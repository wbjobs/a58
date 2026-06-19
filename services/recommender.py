from services.style_rules import (
    STYLE_MATCH_RULES,
    ROOM_ESSENTIAL_FURNITURE,
    get_style_recommendations
)


class FurnitureRecommender:
    def __init__(self):
        self.rules = STYLE_MATCH_RULES
        self.room_essentials = ROOM_ESSENTIAL_FURNITURE

    def generate_recommendations(self, style_result, detection_result):
        style_name = style_result.get("primary_style", "现代简约")
        style_confidence = style_result.get("primary_confidence", 0.0)
        top_k_styles = style_result.get("top_k", [])

        detected_items = detection_result.get("detections", [])
        detected_labels = [item["label"] for item in detected_items]

        style_info = get_style_recommendations(style_name)
        if style_info is None:
            style_info = get_style_recommendations("现代简约")
            style_name = "现代简约"

        existing_items_detail = self._analyze_existing_furniture(
            detected_items, style_info
        )

        missing_items = self._find_missing_furniture(
            detected_labels, style_info
        )

        specific_recommendations = self._generate_specific_recommendations(
            style_name, style_info, missing_items, detected_labels
        )

        room_suggestions = self._suggest_room_essentials(
            detected_labels, style_info
        )

        return {
            "style_analysis": {
                "primary_style": style_name,
                "confidence": style_confidence,
                "style_description": style_info.get("description", ""),
                "color_palette": style_info.get("color_palette", []),
                "recommended_materials": style_info.get("materials", []),
                "alternative_styles": [s for s in top_k_styles if s["style"] != style_name]
            },
            "detected_furniture": existing_items_detail,
            "missing_essential_furniture": missing_items,
            "specific_recommendations": specific_recommendations,
            "room_suggestions": room_suggestions,
            "avoid_items": style_info.get("avoid", [])
        }

    def _analyze_existing_furniture(self, detected_items, style_info):
        results = []
        recommended = style_info.get("recommended_furniture", {})

        for item in detected_items:
            label = item["label"]
            is_match = False
            match_details = None

            if label in recommended:
                is_match = True
                match_details = recommended[label]

            results.append({
                "label": label,
                "confidence": item.get("confidence", 0.0),
                "bbox": item.get("bbox", {}),
                "area": item.get("area", 0),
                "style_match": is_match,
                "recommended_for_style": match_details,
                "style_advice": (
                    "该家具与当前装修风格匹配度高" if is_match
                    else f"建议更换为符合{style_info.get('description', '')}的{label}"
                )
            })

        return results

    def _find_missing_furniture(self, detected_labels, style_info):
        recommended = style_info.get("recommended_furniture", {})
        all_recommended_types = list(recommended.keys())

        missing = []
        for furniture_type in all_recommended_types:
            if furniture_type not in detected_labels:
                missing.append(furniture_type)

        return missing

    def _generate_specific_recommendations(self, style_name, style_info, missing_items, detected_labels):
        recommendations = []
        recommended = style_info.get("recommended_furniture", {})

        for furniture_type in missing_items:
            if furniture_type in recommended:
                rec = recommended[furniture_type]
                recommendations.append({
                    "furniture_type": furniture_type,
                    "priority": "high" if furniture_type in self._get_all_essentials() else "medium",
                    "suggested_style": rec.get("style", ""),
                    "suggested_colors": rec.get("colors", []),
                    "suggested_materials": rec.get("materials", []),
                    "reason": f"为{style_name}空间补充{rec.get('style', furniture_type)}，可以增强整体风格统一感"
                })

        for label in detected_labels:
            if label in recommended:
                rec = recommended[label]
                recommendations.append({
                    "furniture_type": label,
                    "priority": "low",
                    "existing": True,
                    "suggested_style": rec.get("style", ""),
                    "suggested_colors": rec.get("colors", []),
                    "suggested_materials": rec.get("materials", []),
                    "reason": f"已检测到{label}，建议搭配{rec.get('colors', [])}色调的{rec.get('materials', [])}材质周边配饰"
                })

        recommendations.sort(key=lambda x: 0 if x["priority"] == "high" else 1 if x["priority"] == "medium" else 2)
        return recommendations

    def _suggest_room_essentials(self, detected_labels, style_info):
        suggestions = {}
        recommended = style_info.get("recommended_furniture", {})

        for room, essentials in self.room_essentials.items():
            existing = [e for e in essentials if e in detected_labels]
            missing = [e for e in essentials if e not in detected_labels and e in recommended]

            if len(existing) >= 2 or (len(existing) > 0 and len(detected_labels) >= 3):
                suggestions[room] = {
                    "detected": existing,
                    "missing": missing,
                    "completeness": round(len(existing) / len(essentials) * 100, 1),
                    "is_likely_room": True
                }

        if not suggestions:
            suggestions["通用空间"] = {
                "detected": detected_labels,
                "missing": [k for k in recommended.keys() if k not in detected_labels],
                "completeness": 0.0,
                "is_likely_room": False
            }

        return suggestions

    def _get_all_essentials(self):
        essentials = set()
        for items in self.room_essentials.values():
            essentials.update(items)
        return list(essentials)
