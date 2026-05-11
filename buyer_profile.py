from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BuyerProfile:
    name: str
    quiet_importance: int
    school_importance: int
    price_importance: int
    size_importance: int
    lifestyle_importance: int
    min_fraser_score: float
    min_bedrooms: int
    exclude_high_noise: bool
    preferred_city: str
    deal_breakers: tuple[str, ...]

    def defaults(self) -> dict[str, Any]:
        return {
            "quiet_importance": self.quiet_importance,
            "school_importance": self.school_importance,
            "price_importance": self.price_importance,
            "size_importance": self.size_importance,
            "lifestyle_importance": self.lifestyle_importance,
            "min_fraser_score": self.min_fraser_score,
            "min_bedrooms": self.min_bedrooms,
            "exclude_high_noise": self.exclude_high_noise,
            "preferred_city": self.preferred_city,
            "deal_breakers": list(self.deal_breakers),
            "profile_name": self.name,
        }


PRESET_PROFILES: dict[str, BuyerProfile] = {
    "Quiet Family Profile": BuyerProfile(
        name="Quiet Family Profile",
        quiet_importance=5,
        school_importance=4,
        price_importance=3,
        size_importance=4,
        lifestyle_importance=3,
        min_fraser_score=7.0,
        min_bedrooms=3,
        exclude_high_noise=True,
        preferred_city="Both",
        deal_breakers=("Exclude high-noise homes", "Prefer Low/Medium noise streets", "Minimum 3 bedrooms"),
    ),
    "Top School Profile": BuyerProfile(
        name="Top School Profile",
        quiet_importance=3,
        school_importance=5,
        price_importance=2,
        size_importance=3,
        lifestyle_importance=2,
        min_fraser_score=8.5,
        min_bedrooms=3,
        exclude_high_noise=False,
        preferred_city="Both",
        deal_breakers=("Minimum Fraser score 8.5", "School/catchment quality outranks price"),
    ),
    "Value Buyer Profile": BuyerProfile(
        name="Value Buyer Profile",
        quiet_importance=2,
        school_importance=3,
        price_importance=5,
        size_importance=4,
        lifestyle_importance=3,
        min_fraser_score=6.5,
        min_bedrooms=3,
        exclude_high_noise=False,
        preferred_city="Both",
        deal_breakers=("Stay under budget", "Prioritize price fit and usable space"),
    ),
    "Rancher Backyard Profile": BuyerProfile(
        name="Rancher Backyard Profile",
        quiet_importance=5,
        school_importance=2,
        price_importance=3,
        size_importance=4,
        lifestyle_importance=5,
        min_fraser_score=6.5,
        min_bedrooms=3,
        exclude_high_noise=True,
        preferred_city="Both",
        deal_breakers=(
            "Avoid highway/noisy-backyard locations",
            "Prefer rancher or easy main-floor living",
            "Prefer usable private backyard",
            "Value mortgage helper or suite potential",
            "School score matters, but is secondary to home fit",
        ),
    ),
}


def get_profile(name: str) -> BuyerProfile:
    return PRESET_PROFILES.get(name, PRESET_PROFILES["Quiet Family Profile"])


def parse_buyer_profile(text: str) -> dict[str, Any]:
    lowered = text.lower()
    suggestions = {
        "quiet_importance": 3,
        "school_importance": 4,
        "price_importance": 3,
        "size_importance": 3,
        "lifestyle_importance": 3,
        "deal_breakers": [],
    }
    if any(word in lowered for word in ["quiet", "noise", "peaceful", "traffic"]):
        suggestions["quiet_importance"] = 5
        suggestions["deal_breakers"].append("Avoid high-noise homes")
    if any(word in lowered for word in ["school", "fraser", "catchment", "elementary"]):
        suggestions["school_importance"] = 5
    if any(word in lowered for word in ["rancher", "backyard", "yard", "layout", "suite", "mortgage helper"]):
        suggestions["lifestyle_importance"] = 5
    if any(word in lowered for word in ["budget", "value", "affordable", "price"]):
        suggestions["price_importance"] = 5
        suggestions["deal_breakers"].append("Stay under budget")
    if any(word in lowered for word in ["large", "space", "bedroom", "family", "suite"]):
        suggestions["size_importance"] = 5
    return suggestions
