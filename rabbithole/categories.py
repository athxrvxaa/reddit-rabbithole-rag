"""Heuristic topic labels from the AMA title/intro.

This is deliberately simple (keyword rules, first match wins). It's good enough for a UI filter
and for slicing eval results; swap in a zero-shot classifier later if you want better labels.
"""
from __future__ import annotations

import re

_RULES: list[tuple[str, str]] = [
    ("health_medicine", r"doctor|nurse|surgeon|physician|medical|hospital|cancer|therapist|psychiatrist|paramedic|emt|dentist|pharmac\w*|diagnos\w*|disease|surgery|patient|paraplegic|quadriplegic|paralyz\w*|chronic|illness|medication|schizophreni\w*|bipolar|depress\w*|anxiety|ocd|adhd|epilep\w*|diabet\w*|hiv|aids|transplant|blind|deaf|colorblind|amputee|tourette\w*|stutter\w*"),
    ("crime_law", r"police|cop|fbi|cia|nsa|prison|inmate|jail|convict\w*|criminal|felon\w*|arrest\w*|lawyer|attorney|judge|detective|sheriff|dea|gang\w*|murder\w*|heist|thief|burglar|smuggl\w*|prosecut\w*|parole|probation|cartel|correctional|bail"),
    ("military", r"soldier|marine|army|navy|air force|veteran|deploy\w*|combat|sniper|special forces|infantry|military|afghanistan|iraq|vietnam|ranger|submarin\w*"),
    ("science_space", r"scientist|astronaut|nasa|physicist|chemist|biologist|astronom\w*|researcher|phd|geolog\w*|antarctic\w*|meteorolog\w*|archaeolog\w*|paleontolog\w*"),
    ("tech", r"programmer|developer|software|google|microsoft|apple|facebook|amazon|silicon valley|sysadmin|data scientist|engineer\w*|hacker|cyber\w*|tech support|it guy|reddit admin"),
    ("business_finance", r"ceo|founder|startup|entrepreneur|investor|wall street|banker|trader|millionaire|billionaire|rich|wealth\w*|stock\w*|hedge fund|venture|real estate|landlord|salary|accountant|broker|business|lottery"),
    ("entertainment_media", r"actor|actress|director|screenwriter|musician|singer|band|rapper|dj|comedian|youtuber|author|writer|journalist|reporter|producer|hollywood|movie|film|television|tv|model|porn\w*|stripper|escort|cartoon\w*|animator|voice actor"),
    ("gaming_sports", r"gamer|gaming|world of warcraft|wow|video game|esports|streamer|athlete|nfl|nba|mlb|nhl|olympi\w*|football|soccer|baseball|basketball|boxer|mma|ufc|wrestl\w*|coach|poker|chess|tennis|golf|racing|nascar|player"),
    ("education", r"teacher|professor|student|school|college|university|principal|tutor|homeschool\w*|dropout|graduate|teach\w*"),
    ("religion", r"christian|muslim|atheist|jehovah\w*|mormon|catholic|jewish|hindu|buddhist|pastor|priest|preacher|nun|cult|religio\w*|church|scientolog\w*|amish"),
    ("politics_government", r"politic\w*|republican|democrat|senator|congress\w*|mayor|government|election|campaign|lobby\w*|diplomat|activist|libertarian|communist|socialist|conservative|liberal|embassy|tea party"),
    ("jobs_trades", r"driver|trucker|pilot|flight attendant|farmer|plumber|electrician|mechanic|chef|cook|waiter|waitress|bartender|barista|janitor|cashier|construction|welder|carpenter|miner|fisherman|sailor|captain|firefighter|mail ?man|postal|telemarketer|call center|retail|warehouse|security guard|taxi|uber|delivery|locksmith|undertaker|mortician|funeral|embalmer|coroner|garbage|sanitation|repo|stagehand|lifeguard|crossing guard|oil rig|rig"),
    ("travel_adventure", r"travel\w*|backpack\w*|hitchhik\w*|nomad|expat|sail\w*|climb\w*|mountaineer|explorer|world trip|cruise"),
    ("identity_lifestyle", r"gay|lesbian|bisexual|trans|transgender|autis\w*|asperger\w*|virgin|polyamor\w*|adopt\w*|immigrant|refugee|homeless|addict\w*|alcoholic|recovering|sober|vegan|vegetarian|twin|widow\w*|orphan|foster|survivor|abuse\w*|single (?:mom|mother|dad|father)|teen\w*|pregnan\w*|little person|dwarf|midget|centenarian|polygamist"),
    ("relationships_family", r"married|marriage|divorce\w*|husband|wife|girlfriend|boyfriend|mother|father|dad|mom|parent\w*|sister|brother|cheat\w*|affair|dating|relationship|love|wedding|ex-?wife|ex-?husband"),
]

_COMPILED = [(name, re.compile(r"\b(?:" + pat + r")\b", re.I)) for name, pat in _RULES]

CATEGORIES: list[str] = [n for n, _ in _RULES] + ["other"]


def categorize(title: str, context: str = "") -> str:
    # Title is the strongest signal; fall back to the intro text.
    for text in (title or "", context or ""):
        if not text:
            continue
        for name, rx in _COMPILED:
            if rx.search(text):
                return name
    return "other"
