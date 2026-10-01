"""Transparent rule matchers for region (country/city) and tracked role. Rules live in config/."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from xray.config import RegionRule, SourcesConfig


def _phrase_regex(phrases: Iterable[str]) -> re.Pattern[str]:
    """Case-insensitive whole-phrase alternation; boundaries are 'not alphanumeric'."""
    alts = sorted(
        {re.escape(p.strip().lower()) for p in phrases if p.strip()}, key=len, reverse=True
    )
    return re.compile(rf"(?<![a-z0-9])(?:{'|'.join(alts)})(?![a-z0-9])", re.I)


@dataclass(frozen=True)
class RegionMatch:
    country: str
    evidence: str
    cities: tuple[str, ...]


class RegionMatcher:
    def __init__(self, rules: Mapping[str, RegionRule]) -> None:
        self._rules = [
            (
                key,
                rule,
                _phrase_regex(rule.country_aliases),
                [(city, _phrase_regex(aliases)) for city, aliases in rule.cities.items()],
            )
            for key, rule in rules.items()
        ]

    def match(self, locations: list[str], structured_countries: list[str]) -> list[RegionMatch]:
        out = []
        for key, rule, country_rx, city_rxs in self._rules:
            cities = tuple(c for c, rx in city_rxs if any(rx.search(loc) for loc in locations))
            evidence = next(
                (
                    f"structured_country={sc}"
                    for sc in structured_countries
                    if sc.strip().upper() == rule.iso2 or country_rx.fullmatch(sc.strip())
                ),
                None,
            )
            if evidence is None:
                evidence = next(
                    (f"location~{loc}" for loc in locations if country_rx.search(loc)), None
                )
            if evidence is None and cities:
                evidence = next(
                    f"city~{loc}" for loc in locations for _, rx in city_rxs if rx.search(loc)
                )
            if evidence is not None:
                out.append(RegionMatch(key, evidence, cities))
        return out


class RoleMatcher:
    def __init__(self, roles: Mapping[str, list[str]], exclude: Iterable[str] = ()) -> None:
        self._roles = [(role, _phrase_regex(pats)) for role, pats in roles.items()]
        exclude = [e for e in exclude if e.strip()]
        self._exclude = _phrase_regex(exclude) if exclude else None

    @property
    def names(self) -> list[str]:
        return [role for role, _ in self._roles]

    @classmethod
    def from_config(cls, sources: SourcesConfig) -> RoleMatcher:
        return cls(sources.roles_to_track, sources.role_title_exclude)

    def match(self, title: str) -> list[str]:
        if self._exclude is not None and self._exclude.search(title):
            return []
        return [role for role, rx in self._roles if rx.search(title)]
