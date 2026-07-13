from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


AGENTS = ("cortex", "executor", "mouth", "mimicry", "rem", "fleet")
APPROVED_LICENSES = frozenset({"Apache-2.0", "CC-BY-4.0", "MIT"})
ML_TRAINING_PARTITIONS = frozenset({"train"})
PARTITION_KINDS = frozenset({"ml_split", "reference_corpus"})
ACCESS_MODES = frozenset({"public_https", "local_override"})
REDISTRIBUTION_MODES = frozenset({"transformed_records_only"})
QUALITY_PROFILES = frozenset(
    {
        "human_dialogue_grounding",
        "human_intent_annotations",
        "human_meaning_preserving_edits",
        "standards_conformance_cases",
        "verified_synthetic_tool_use",
    }
)
TRANSFORMERS = frozenset(
    {
        "apigen_xlam_v1",
        "coedit_v1",
        "faithdial_v1",
        "json_schema_tests_v1",
        "massive_v1",
        "oasst2_v2",
        "toolace_v1",
    }
)
SOURCE_MANIFEST_PATH = Path(__file__).with_name("public_adapter_corpus_sources.json")
SOURCE_MANIFEST_SCHEMA = "lumen.public-adapter-corpus-sources/1.1.0"
SNAPSHOT_SCHEMA = "lumen.public-adapter-corpus/1.1.0"
RECORD_SCHEMA = "lumen.public-adapter-corpus-record/1.1.0"
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_TEXT_CHARS = 4_000
MIN_TEXT_CHARS = 2

_EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){7,15}(?!\w)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PAYMENT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_UUID_RE = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\[\]{}()]+")
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)*|\.\d+)(?!\w)")
_NEGATION_RE = re.compile(r"(?i)\b(?:no|not|never|neither|nor|without|cannot|can't|won't|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|shouldn't|wouldn't|couldn't)\b")
_CODE_RE = re.compile(r"(?i)(?:```|<\/?(?:script|style|html)\b|\b(?:def|class|function)\s+[A-Za-z_]\w*\s*\(|\b(?:SELECT|INSERT|UPDATE|DELETE)\s+\w+\s+(?:FROM|INTO|SET)\b)")
_HIGH_STAKES_RE = re.compile(
    r"(?i)\b(?:"
    r"abortion|adhd|allerg(?:y|ies|ic)|asthma(?:tic)?|attorney|autis(?:m|tic)|autoimmune|bankruptcy|"
    r"blood pressure|cancer|clinical|contract law|court|credit score|debt|depress(?:ion|ed)|"
    r"diagnos\w*|disease\w*|doctor|dosage|dose|drug\w*|"
    r"emergency|financial advice|healthcare|insurance|invest(?:ing|ment)|lawyer|lawsuit|legal advice|"
    r"illness(?:es)?|infection\w*|medical|medication\w*|mental health|mortgage|patient|pharmaceutical\w*|"
    r"prescri\w*|retirement account|self[- ]?harm|surgery|symptom\w*|syndrome\w*|tax advice|"
    r"therap(?:ist|ists|y|ies)|treatment\w*|tumou?r\w*|vaccin\w*"
    r")\b"
)
_CURRENT_RECOMMENDATION_RE = re.compile(
    r"(?i)\b(?:best|current|deal|latest|near me|newest|price|product recommendation|"
    r"recommend(?:ation|ed)?|restaurant recommendation|right now|today'?s|top rated|travel itinerary|"
    r"up[- ]to[- ]date|which (?:one|product|phone|laptop|car) should i buy|yesterday|recently|this week)\b"
)
_SECURITY_RE = re.compile(
    r"(?i)\b(?:api key|authentication|authorization|cyber(?:security)?|ddos|decrypt|encrypt(?:ion)?|"
    r"exploit|firewall|hack(?:er|ing)?|malware|password|penetration test|phishing|ransomware|"
    r"security token|sql injection|system prompt|vulnerabilit(?:y|ies)|xss)\b"
)
_MODEL_IDENTITY_RE = re.compile(
    r"(?i)\b(?:as an ai|artificial intelligence|chatgpt|language model|open[ -]?assistant|"
    r"system message|system prompt|what are you|who are you|your (?:creator|developers?|makers?))\b"
)
_COPYRIGHT_STYLE_RE = re.compile(
    r"(?i)(?:\b(?:copyright(?:ed)?|fan fiction|lyrics|song lyrics|transcrib(?:e|ing|ed)|verbatim)\b|"
    r"\b(?:imitate|mimic|copy)\s+(?:the\s+)?(?:style|voice|writing)\b|"
    r"\b(?:in|using)\s+(?:the\s+)?(?:artistic\s+|writing\s+)?style\s+of\b|"
    r"\b(?:write|sound)\s+like\s+[A-Z][A-Za-z'’-]+)"
)
_UNSAFE_EDIT_RE = re.compile(r"(?i)\b(?:fuck(?:ed|ing)?|shit(?:ty)?|suicide|heroin|cocaine|porn(?:ography)?)\b")
_CORPUS_ARTIFACT_RE = re.compile(
    r"(?i)(?:<\s*SEP\s*>|\bJump\s+up\b|\[\s*(?:citation needed|\d{1,3})\s*\]|"
    r"(?:^|\s)\^\s*(?:p\.?\s*\d+|[A-Z][A-Za-z'’-]+))"
)
_META_RESPONSE_RE = re.compile(
    r"(?i)^(?:certainly|here (?:are|is)|i hope (?:this|that)|let me|of course|sure)[,!:;\s]"
)
_QUESTION_LEAD_RE = re.compile(
    r"(?i)^\s*(?:"
    r"who|what|where|when|why|how|which|whose|whom|can|could|would|should|is|are|was|were|"
    r"do|does|did|will|have|has|"
    r"qui|que|qu['’]|quoi|où|quand|pourquoi|comment|quel(?:le|les|s)?|est[- ]ce|"
    r"peut[- ]on|pourriez[- ]vous|serait[- ]il|dois[- ]je"
    r")\b"
)
_QUESTION_FRAGMENT_RE = re.compile(
    r"(?i)^\s*(?:the question (?:is|of)|whether|what about|how about|"
    r"la question (?:est|de)|qu['’]en est[- ]il|et si)\b"
)
_NON_ANSWER_RE = re.compile(
    r"(?i)(?:^|\b)(?:"
    r"i(?:'m| am) sorry|i (?:do not|don['’]t) know|not sure|please clarify|"
    r"need more (?:details|information)|(?:cannot|can['’]t) answer|"
    r"je suis désolé|je ne sais pas|pas certain|veuillez préciser|"
    r"besoin de plus (?:de )?(?:détails|informations)|impossible (?:de|d['’])répondre"
    r")(?:\b|[,.:;])"
)
_DANGLING_END_RE = re.compile(
    r"(?i)\b(?:a|an|and|as|at|because|but|by|for|from|if|in|of|on|or|that|the|to|when|which|with)$"
)
_NUMBER_WORD_RE = re.compile(
    r"(?i)\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundred|thousand|million|billion|first|second|third|fourth|fifth|"
    r"sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|"
    r"sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|"
    r"sixtieth|seventieth|eightieth|ninetieth|hundredth|thousandth)\b"
)
_DIGIT_ORDINAL_RE = re.compile(r"(?i)(?<!\w)\d+(?:st|nd|rd|th)(?!\w)")
_MEASUREMENT_UNIT_RE = re.compile(
    r"(?i)(?:(?<!\w)%(?!\w)|(?<!\w)(?:percent|percentage|degrees?|°[cf]|celsius|fahrenheit|millimeters?|centimeters?|"
    r"meters?|kilometers?|inches?|feet|foot|yards?|miles?|milligrams?|grams?|kilograms?|ounces?|"
    r"pounds?|milliliters?|liters?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|mm|cm|km|mg|"
    r"kg|ml|mph|kph|km/h|"
    r"m/s|hz|khz|mhz|ghz|kb|mb|gb|tb|volts?|watts?)\b)"
)
_TEMPORAL_FACT_RE = re.compile(
    r"(?i)\b(?:today|tomorrow|yesterday|currently|recently|this (?:week|month|year)|"
    r"last (?:week|month|year)|next (?:week|month|year))\b"
)
_MALFORMED_TEXT_RE = re.compile(
    r"(?i)(?:[\"']\s*[\"']|\b(?:over|about|from|between)\s+(?:while|and|[,.!?])|"
    r"\b[A-Za-z]\s+[A-Za-z](?:\s+[A-Za-z]){2,}\b)"
)
_WAKE_WORD_RE = re.compile(r"(?i)^(?:(?:hey|ok|okay)\s+)?(?:alexa|olly)\b[\s,:-]*")
_SLOT_RE = re.compile(r"\[\s*([^]:]+?)\s*:\s*([^\]]+?)\s*\]")
_INSTRUCTION_PREFIX_RE = re.compile(
    r"(?i)^(?:remove all grammatical errors from this text|improve the grammaticality(?: of this sentence)?|"
    r"improve the grammar of this text|fix grammaticality in this sentence|fix grammar in the sentence|"
    r"update to remove grammar errors|make this (?:text|sentence) (?:clearer|more coherent|simpler|neutral)|"
    r"rewrite (?:this|the following).{0,80}?)\s*:\s*"
)
_ROLE_PROMPT_ARTIFACT_RE = re.compile(
    r"(?i)(?:<\/?(?:phy|user|assistant)>|\bRole definition:\s*|\bHistorical dialog data is as follows:)"
)
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9'’-]*")

_QUALITY_SOURCE_CONFIDENCE = {
    "human_dialogue_grounding": 1.0,
    "human_intent_annotations": 0.95,
    "human_meaning_preserving_edits": 0.95,
    "standards_conformance_cases": 1.0,
    "verified_synthetic_tool_use": 0.8,
}

_FACTUAL_STATUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "falsehood",
        re.compile(
            r"(?i)\b(?:false|untrue|incorrect|inaccurate|debunk(?:s|ed|ing)?|"
            r"disprov(?:e|es|ed|en|ing))\b"
        ),
    ),
    (
        "truth",
        re.compile(r"(?i)\b(?:true|correct|accurate|verified|proven|factual)\b"),
    ),
    (
        "attribution",
        re.compile(
            r"(?i)(?:\b(?:alleg(?:e|es|ed|ing|edly)|purport(?:s|ed|ing|edly)?|"
            r"suppos(?:e|es|ed|ing|edly)|claim(?:s|ed|ing)?|assert(?:s|ed|ing|ion)?|"
            r"indicat(?:e|es|ed|ing)|attribut(?:e|es|ed|ing|ion)|cit(?:e|es|ed|ing)|"
            r"say|says|said|saying|state|states|stated|stating)\b|\baccording to\b)"
        ),
    ),
    (
        "denial_or_dispute",
        re.compile(
            r"(?i)\b(?:den(?:y|ies|ied|ying)|reject(?:s|ed|ing|ion)?|"
            r"disput(?:e|es|ed|ing)|question(?:s|ed|ing)?)\b"
        ),
    ),
    (
        "confirmation",
        re.compile(
            r"(?i)\b(?:confirm(?:s|ed|ing|ation)?|corroborat(?:e|es|ed|ing|ion)|"
            r"substantiat(?:e|es|ed|ing|ion)|validat(?:e|es|ed|ing|ion))\b"
        ),
    ),
    (
        "accusation_or_blame",
        re.compile(
            r"(?i)\b(?:accus(?:e|es|ed|ing|ation)|blam(?:e|es|ed|ing)|"
            r"ascrib(?:e|es|ed|ing))\b"
        ),
    ),
    (
        "uncertainty",
        re.compile(
            r"(?i)\b(?:may|might|could|possibly|perhaps|probably|likely|unlikely|unclear|uncertain|"
            r"unverified|unproven|theoretical(?:ly)?|suspect(?:s|ed|ing)?)\b"
        ),
    ),
    (
        "certainty",
        re.compile(r"(?i)\b(?:certainly|definitely|clearly|undoubtedly|conclusively)\b"),
    ),
    (
        "normative_requirement",
        re.compile(
            r"(?i)(?:\b(?:should|must|shall)(?!\s+not\b)|\bought\s+to\b|"
            r"\b(?:requir(?:e|es|ed|ing|ement)|mandat(?:e|es|ed|ing|ory)|"
            r"oblig(?:e|es|ed|ing|ation|atory))\b|\bneeds?\s+to\b)"
        ),
    ),
    (
        "normative_permission",
        re.compile(r"(?i)\b(?:permit(?:s|ted|ting)?|allow(?:s|ed|ing)?|authoriz(?:e|es|ed|ing))\b"),
    ),
    (
        "normative_prohibition",
        re.compile(
            r"(?i)(?:\b(?:should|must|shall)\s+not\b|\bnot\s+(?:permitted|allowed|authorized)\b|"
            r"\b(?:prohibit(?:s|ed|ing)?|forbid(?:s|den|ding)?|banned|barred|disallowed)\b)"
        ),
    ),
)


class PublicCorpusError(ValueError):
    """Raised when a public corpus source or snapshot violates its contract."""


@dataclass(frozen=True)
class PublicCorpusBuildResult:
    output_dir: Path
    records_path: Path
    manifest_path: Path
    attribution_path: Path
    record_count: int
    counts_by_agent: dict[str, int]
    records_sha256: str


@dataclass(frozen=True)
class LumenContract:
    sha256: str
    tools: dict[str, dict[str, Any]]
    intents: dict[str, dict[str, Any]]
    cortex_required_fields: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_content_sha256(value: Any) -> str:
    def encode_unknown(item: Any) -> str:
        isoformat = getattr(item, "isoformat", None)
        if callable(isoformat):
            return str(isoformat())
        if isinstance(item, bytes):
            return item.hex()
        raise TypeError(f"Unsupported source value for canonical hashing: {type(item).__name__}")

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=encode_unknown,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_public_corpus_source_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or SOURCE_MANIFEST_PATH
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicCorpusError(f"Unable to read public corpus source manifest {manifest_path}: {error}") from error
    _validate_source_manifest(payload)
    return payload


def _validate_source_manifest(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise PublicCorpusError("Unsupported public corpus source manifest schema")
    policy = payload.get("selectionPolicyVersion")
    if not isinstance(policy, str) or not policy.strip():
        raise PublicCorpusError("selectionPolicyVersion must be a non-empty string")
    allowed = payload.get("allowedLicenses")
    if not isinstance(allowed, list) or not allowed or any(not isinstance(item, str) for item in allowed):
        raise PublicCorpusError("allowedLicenses must be a non-empty string list")
    if not set(allowed) <= APPROVED_LICENSES:
        raise PublicCorpusError("allowedLicenses contains a license outside Lumen's approved public-corpus policy")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PublicCorpusError("sources must be a non-empty list")
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise PublicCorpusError("Every source must be an object")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise PublicCorpusError(f"Invalid or duplicate source id: {source_id!r}")
        seen.add(source_id)
        if source.get("license") not in allowed:
            raise PublicCorpusError(f"Source {source_id} has a disallowed or unknown license")
        for key in (
            "datasetID",
            "revision",
            "sourceURL",
            "artifactFormat",
            "artifactSHA256",
            "partitionKind",
            "sourcePartition",
            "transformer",
            "qualityProfile",
            "accessMode",
            "redistributionMode",
        ):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise PublicCorpusError(f"Source {source_id} requires {key}")
        if source["transformer"] not in TRANSFORMERS:
            raise PublicCorpusError(f"Source {source_id} has unsupported transformer {source['transformer']!r}")
        if source["qualityProfile"] not in QUALITY_PROFILES:
            raise PublicCorpusError(f"Source {source_id} has unsupported qualityProfile {source['qualityProfile']!r}")
        if source["accessMode"] not in ACCESS_MODES:
            raise PublicCorpusError(f"Source {source_id} has unsupported accessMode {source['accessMode']!r}")
        if source["redistributionMode"] not in REDISTRIBUTION_MODES:
            raise PublicCorpusError(
                f"Source {source_id} has unsupported redistributionMode {source['redistributionMode']!r}"
            )
        partition_kind = source["partitionKind"]
        source_partition = source["sourcePartition"].strip().lower()
        if partition_kind not in PARTITION_KINDS:
            raise PublicCorpusError(f"Source {source_id} has unsupported partitionKind {partition_kind!r}")
        if partition_kind == "ml_split" and source_partition not in ML_TRAINING_PARTITIONS:
            raise PublicCorpusError(
                f"Source {source_id} ML partition {source_partition!r} is not approved for training"
            )
        for key in ("sourceURL", "licenseURL"):
            if not isinstance(source.get(key), str) or not source[key].startswith("https://"):
                raise PublicCorpusError(f"Source {source_id} requires an HTTPS {key}")
        artifact_url = source.get("artifactURL")
        if source["accessMode"] == "public_https":
            if not isinstance(artifact_url, str) or not artifact_url.startswith("https://"):
                raise PublicCorpusError(f"Source {source_id} requires an HTTPS artifactURL")
        elif artifact_url is not None and (
            not isinstance(artifact_url, str) or not artifact_url.startswith("https://")
        ):
            raise PublicCorpusError(f"Source {source_id} artifactURL must be HTTPS when present")
        if not isinstance(source.get("attribution"), str) or not source["attribution"].strip():
            raise PublicCorpusError(f"Source {source_id} requires attribution")
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise PublicCorpusError(f"Source {source_id} revision must be a full 40-character commit SHA")
        if not re.fullmatch(r"[0-9a-f]{64}", source["artifactSHA256"]):
            raise PublicCorpusError(f"Source {source_id} artifactSHA256 must be a lowercase SHA-256")
        artifact_format = source["artifactFormat"]
        if artifact_format in {"json", "jsonl", "parquet"}:
            source_path = source.get("artifactPath")
            if not isinstance(source_path, str) or not source_path or source_path.startswith(("/", "../")):
                raise PublicCorpusError(f"Source {source_id} requires a safe relative artifactPath")
        elif artifact_format == "tar.gz-jsonl":
            member = source.get("artifactMember")
            if not isinstance(member, str) or not member or member.startswith(("/", "../")):
                raise PublicCorpusError(f"Source {source_id} requires a safe relative artifactMember")
        elif artifact_format == "tar.gz-json":
            member_prefix = source.get("artifactMemberPrefix")
            selected_files = source.get("selectedFiles")
            if (
                not isinstance(member_prefix, str)
                or not member_prefix
                or member_prefix.startswith(("/", "../"))
                or not isinstance(selected_files, list)
                or not selected_files
                or any(
                    not isinstance(selected_file, str)
                    or not selected_file
                    or selected_file.startswith(("/", "../"))
                    or ".." in Path(selected_file).parts
                    for selected_file in selected_files
                )
            ):
                raise PublicCorpusError(f"Source {source_id} requires a safe member prefix and selectedFiles")
        else:
            raise PublicCorpusError(f"Source {source_id} has unsupported artifactFormat {artifact_format}")
        caps = source.get("adapterCaps")
        if not isinstance(caps, dict) or not caps:
            raise PublicCorpusError(f"Source {source_id} requires adapterCaps")
        for agent, cap in caps.items():
            if agent not in AGENTS or type(cap) is not int or cap <= 0:
                raise PublicCorpusError(f"Source {source_id} has invalid adapter cap {agent}={cap!r}")
        target_adapters = source.get("targetAdapters")
        if (
            not isinstance(target_adapters, list)
            or not target_adapters
            or any(agent not in AGENTS for agent in target_adapters)
            or len(set(target_adapters)) != len(target_adapters)
            or set(target_adapters) != set(caps)
        ):
            raise PublicCorpusError(
                f"Source {source_id} targetAdapters must exactly match adapterCaps"
            )
        if source["transformer"] == "apigen_xlam_v1":
            _validate_apigen_xlam_source_contract(source)


def _validate_apigen_xlam_source_contract(source: Mapping[str, Any]) -> None:
    source_id = source["id"]
    if source["accessMode"] != "local_override":
        raise PublicCorpusError(
            f"Source {source_id} uses apigen_xlam_v1 and must use local_override access"
        )
    if source["artifactFormat"] not in {"json", "jsonl"}:
        raise PublicCorpusError(
            f"Source {source_id} uses apigen_xlam_v1 and requires a JSON or JSONL artifact"
        )
    if source["qualityProfile"] != "verified_synthetic_tool_use":
        raise PublicCorpusError(
            f"Source {source_id} uses apigen_xlam_v1 and requires verified_synthetic_tool_use quality"
        )
    if not set(source["targetAdapters"]) <= {"cortex", "executor"}:
        raise PublicCorpusError(
            f"Source {source_id} uses apigen_xlam_v1 and may target only cortex and executor"
        )
    mappings = source.get("toolMappings")
    if not isinstance(mappings, dict) or not mappings:
        raise PublicCorpusError(
            f"Source {source_id} uses apigen_xlam_v1 and requires explicit toolMappings"
        )
    for upstream_name, mapping in mappings.items():
        if not isinstance(upstream_name, str) or not upstream_name.strip() or not isinstance(mapping, dict):
            raise PublicCorpusError(f"Source {source_id} has an invalid APIGen/xLAM tool mapping")
        allowed_keys = {"argumentMappings", "intent", "toolID"}
        if set(mapping) - allowed_keys:
            raise PublicCorpusError(
                f"Source {source_id} mapping {upstream_name!r} contains unsupported fields"
            )
        tool_id = mapping.get("toolID")
        argument_mappings = mapping.get("argumentMappings")
        if (
            not isinstance(tool_id, str)
            or not tool_id.strip()
            or not isinstance(argument_mappings, dict)
            or any(
                not isinstance(source_argument, str)
                or not source_argument.strip()
                or not isinstance(target_argument, str)
                or not target_argument.strip()
                for source_argument, target_argument in argument_mappings.items()
            )
            or len(set(argument_mappings.values())) != len(argument_mappings)
        ):
            raise PublicCorpusError(
                f"Source {source_id} mapping {upstream_name!r} requires an exact argumentMappings contract"
            )
        intent = mapping.get("intent")
        if "cortex" in source["targetAdapters"] and (
            not isinstance(intent, str) or not intent.strip()
        ):
            raise PublicCorpusError(
                f"Source {source_id} mapping {upstream_name!r} requires intent when Cortex is targeted"
            )
        if intent is not None and (not isinstance(intent, str) or not intent.strip()):
            raise PublicCorpusError(
                f"Source {source_id} mapping {upstream_name!r} has an invalid intent"
            )


def load_lumen_contract(path: Path) -> LumenContract:
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise PublicCorpusError(f"Invalid Lumen manifest JSON at {path}: {error}") from error
    return lumen_contract_from_manifest(payload)


def lumen_contract_from_manifest(manifest: Any) -> LumenContract:
    payload = manifest.model_dump() if callable(getattr(manifest, "model_dump", None)) else manifest
    if not isinstance(payload, dict):
        raise PublicCorpusError("Lumen manifest must be a JSON object")
    tools = payload.get("tools")
    intents = payload.get("intents")
    if not isinstance(tools, list) or not isinstance(intents, list):
        raise PublicCorpusError("Lumen manifest requires tools and intents arrays")
    agent_protocols = payload.get("agentProtocols")
    cortex_output = agent_protocols.get("cortexOutput") if isinstance(agent_protocols, dict) else None
    required_fields = cortex_output.get("requiredFields") if isinstance(cortex_output, dict) else None
    native_cortex_fields = {"intent", "selectedToolID", "requiresApproval", "nextModel", "reasoningSummary"}
    if (
        not isinstance(required_fields, list)
        or any(not isinstance(field, str) for field in required_fields)
        or not native_cortex_fields.issubset(required_fields)
    ):
        raise PublicCorpusError("Lumen manifest Cortex output protocol is missing native required fields")
    tool_map: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("id"), str):
            raise PublicCorpusError("Lumen manifest contains an invalid tool")
        if tool["id"] in tool_map:
            raise PublicCorpusError(f"Lumen manifest contains duplicate tool {tool['id']}")
        tool_map[tool["id"]] = tool
    intent_map: dict[str, dict[str, Any]] = {}
    for intent in intents:
        if not isinstance(intent, dict) or not isinstance(intent.get("id"), str):
            raise PublicCorpusError("Lumen manifest contains an invalid intent")
        if intent["id"] in intent_map:
            raise PublicCorpusError(f"Lumen manifest contains duplicate intent {intent['id']}")
        intent_map[intent["id"]] = intent
    if not tool_map or not intent_map:
        raise PublicCorpusError("Lumen manifest tool and intent catalogs cannot be empty")
    canonical_contract = {
        "schema": "lumen.public-adapter-corpus-contract/1.0.0",
        "cortexRequiredFields": sorted(set(required_fields)),
        "tools": [_canonical_tool_contract(tool_map[tool_id]) for tool_id in sorted(tool_map)],
        "intents": [_canonical_intent_contract(intent_map[intent_id]) for intent_id in sorted(intent_map)],
    }
    return LumenContract(
        sha256=_sha256_bytes(_canonical_json(canonical_contract).encode("utf-8")),
        tools=tool_map,
        intents=intent_map,
        cortex_required_fields=tuple(required_fields),
    )


def _canonical_tool_contract(tool: Mapping[str, Any]) -> dict[str, Any]:
    arguments: list[dict[str, Any]] = []
    raw_arguments = tool.get("arguments")
    if not isinstance(raw_arguments, list):
        raise PublicCorpusError(f"Lumen tool {tool.get('id')} arguments must be an array")
    for argument in raw_arguments:
        if not isinstance(argument, dict) or not isinstance(argument.get("name"), str):
            raise PublicCorpusError(f"Lumen tool {tool.get('id')} contains an invalid argument")
        allowed_values = argument.get("allowedValues")
        arguments.append(
            {
                "allowedValues": sorted(allowed_values) if isinstance(allowed_values, list) else None,
                "name": argument["name"],
                "required": argument.get("required") is not False,
                "type": argument.get("type"),
            }
        )
    return {
        "arguments": sorted(arguments, key=lambda item: item["name"]),
        "confirmationMode": tool.get("confirmationMode"),
        "id": tool["id"],
        "permissionKind": tool.get("permissionKind"),
        "requiresApproval": tool.get("requiresApproval") is True,
    }


def _canonical_intent_contract(intent: Mapping[str, Any]) -> dict[str, Any]:
    allowed_tool_ids = intent.get("allowedToolIDs")
    if allowed_tool_ids is None:
        allowed_tool_ids = []
    if not isinstance(allowed_tool_ids, list) or any(not isinstance(item, str) for item in allowed_tool_ids):
        raise PublicCorpusError(f"Lumen intent {intent.get('id')} allowedToolIDs must be a string array")
    return {"allowedToolIDs": sorted(set(allowed_tool_ids)), "id": intent["id"]}


def acquire_public_corpus_sources(
    cache_dir: Path,
    *,
    source_manifest_path: Path | None = None,
    offline: bool = False,
    artifact_paths: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    source_manifest = load_public_corpus_source_manifest(source_manifest_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    overrides = dict(artifact_paths or {})
    resolved: dict[str, Path] = {}
    for source in source_manifest["sources"]:
        source_id = source["id"]
        expected = source["artifactSHA256"]
        if source_id in overrides:
            candidate = Path(overrides[source_id])
        else:
            candidate = cache_dir / _artifact_cache_filename(source)
        if candidate.is_file():
            actual = _sha256_file(candidate)
            if actual != expected:
                if source_id in overrides or offline:
                    raise PublicCorpusError(
                        f"Artifact hash mismatch for {source_id}: expected {expected}, found {actual}"
                    )
                candidate.unlink()
            else:
                resolved[source_id] = candidate
                continue
        if source_id in overrides:
            raise PublicCorpusError(f"Missing artifact override for {source_id}: {candidate}")
        if source["accessMode"] == "local_override":
            raise PublicCorpusError(
                f"Source {source_id} is gated and requires a hash-verified local artifact override"
            )
        if offline:
            raise PublicCorpusError(f"Offline build is missing verified artifact for {source_id}: {candidate}")
        _download_verified(source["artifactURL"], candidate, expected)
        resolved[source_id] = candidate
    return resolved


def _artifact_cache_filename(source: Mapping[str, Any]) -> str:
    suffix = {
        "json": ".json",
        "jsonl": ".jsonl",
        "jsonl.gz": ".jsonl.gz",
        "parquet": ".parquet",
        "tar.gz-jsonl": ".tar.gz",
        "tar.gz-json": ".tar.gz",
    }.get(str(source["artifactFormat"]), ".artifact")
    return f"{source['id']}-{str(source['artifactSHA256'])[:16]}{suffix}"


def _download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "LumenPublicAdapterCorpus/1.0"})
    digest = hashlib.sha256()
    total = 0
    temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".part",
            delete=False,
        ) as output, urllib.request.urlopen(request, timeout=60) as response:
            temp = Path(output.name)
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_ARTIFACT_BYTES:
                    raise PublicCorpusError(f"Artifact exceeds {MAX_ARTIFACT_BYTES} bytes: {url}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise PublicCorpusError(f"Downloaded artifact hash mismatch: expected {expected_sha256}, found {actual}")
        if temp is None:
            raise PublicCorpusError(f"Download did not create a temporary artifact: {url}")
        os.replace(temp, destination)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def build_public_adapter_corpus(
    output_dir: Path,
    *,
    cache_dir: Path,
    lumen_manifest_path: Path,
    source_manifest_path: Path | None = None,
    offline: bool = False,
    artifact_paths: Mapping[str, Path] | None = None,
) -> PublicCorpusBuildResult:
    source_manifest_path = source_manifest_path or SOURCE_MANIFEST_PATH
    source_manifest = load_public_corpus_source_manifest(source_manifest_path)
    source_manifest_raw = source_manifest_path.read_bytes()
    contract = load_lumen_contract(lumen_manifest_path)
    artifacts = acquire_public_corpus_sources(
        cache_dir,
        source_manifest_path=source_manifest_path,
        offline=offline,
        artifact_paths=artifact_paths,
    )
    records: list[dict[str, Any]] = []
    for source in source_manifest["sources"]:
        source_records = _transform_source(
            source,
            artifacts[source["id"]],
            source_manifest["selectionPolicyVersion"],
            contract,
        )
        records.extend(source_records)
    records.sort(key=lambda item: (item["metadata"]["agent"], item["metadata"]["publicCorpus"]["sourceID"], item["id"]))
    _validate_records(records, source_manifest, contract)

    records_bytes = b"".join((_canonical_json(record) + "\n").encode("utf-8") for record in records)
    records_sha = _sha256_bytes(records_bytes)
    counts_by_agent = {agent: 0 for agent in AGENTS}
    counts_by_source: Counter[str] = Counter()
    counts_by_task_type: Counter[str] = Counter()
    scores_by_agent: dict[str, list[float]] = defaultdict(list)
    for record in records:
        agent = record["metadata"]["agent"]
        counts_by_agent[agent] += 1
        counts_by_source[record["metadata"]["publicCorpus"]["sourceID"]] += 1
        counts_by_task_type[record["taskType"]] += 1
        scores_by_agent[agent].append(record["metadata"]["publicCorpus"]["selectionScore"]["overall"])
    counts_by_agent = {agent: count for agent, count in counts_by_agent.items() if count}
    snapshot_manifest = {
        "schema": SNAPSHOT_SCHEMA,
        "sourceManifestSchema": source_manifest["schema"],
        "selectionPolicyVersion": source_manifest["selectionPolicyVersion"],
        "sourceManifestSHA256": _sha256_bytes(source_manifest_raw),
        "lumenContractSHA256": contract.sha256,
        "partitionPolicy": {
            "allowedMLTrainingPartitions": sorted(ML_TRAINING_PARTITIONS),
            "heldoutMLPartitionsExcluded": True,
            "referenceCorporaRequireExplicitPartitionKind": True,
        },
        "recordCount": len(records),
        "recordsFile": "records.jsonl",
        "recordsSHA256": records_sha,
        "countsByAgent": counts_by_agent,
        "countsBySource": dict(sorted(counts_by_source.items())),
        "countsByTaskType": dict(sorted(counts_by_task_type.items())),
        "qualityScoreSummaryByAgent": {
            agent: {
                "maximum": max(scores),
                "mean": round(sum(scores) / len(scores), 6),
                "minimum": min(scores),
            }
            for agent, scores in sorted(scores_by_agent.items())
        },
        "allowedLicenses": source_manifest["allowedLicenses"],
        "sources": source_manifest["sources"],
        "privacy": {
            "piiRegexScreened": True,
            "rawOpenAssistantIdentifiersPersisted": False,
            "rawSourceArtifactsIncluded": False,
        },
    }
    manifest_bytes = (json.dumps(snapshot_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    attribution_bytes = _attribution_markdown(source_manifest).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / "records.jsonl", records_bytes)
    _atomic_write(output_dir / "manifest.json", manifest_bytes)
    _atomic_write(output_dir / "THIRD_PARTY_DATASETS.md", attribution_bytes)
    return PublicCorpusBuildResult(
        output_dir=output_dir,
        records_path=output_dir / "records.jsonl",
        manifest_path=output_dir / "manifest.json",
        attribution_path=output_dir / "THIRD_PARTY_DATASETS.md",
        record_count=len(records),
        counts_by_agent=counts_by_agent,
        records_sha256=records_sha,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temp = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _transform_source(
    source: dict[str, Any],
    artifact: Path,
    policy_version: str,
    contract: LumenContract,
) -> list[dict[str, Any]]:
    transformer = source["transformer"]
    if transformer == "apigen_xlam_v1":
        records = _transform_apigen_xlam(source, artifact, policy_version, contract)
    elif transformer == "massive_v1":
        records = _transform_massive(source, artifact, policy_version, contract)
    elif transformer == "oasst2_v2":
        records = _transform_oasst2(source, artifact, policy_version)
    elif transformer == "coedit_v1":
        records = _transform_coedit(source, artifact, policy_version)
    elif transformer == "json_schema_tests_v1":
        records = _transform_json_schema_tests(source, artifact, policy_version)
    elif transformer == "toolace_v1":
        records = _transform_toolace(source, artifact, policy_version, contract)
    elif transformer == "faithdial_v1":
        records = _transform_faithdial(source, artifact, policy_version)
    else:  # Manifest validation should make this unreachable.
        raise PublicCorpusError(f"No transformer is registered for source {source['id']}")
    return _score_and_select_source_records(records, source)


_MASSIVE_LUMEN_INTENTS: dict[str, str] = {
    "alarm_query": "alarm",
    "alarm_remove": "alarm",
    "alarm_set": "alarm",
    "calendar_query": "calendar",
    "calendar_set": "calendar",
    "cooking_query": "webSearch",
    "cooking_recipe": "webSearch",
    "datetime_convert": "chat",
    "datetime_query": "chat",
    "email_query": "outlook",
    "email_querycontact": "contactSearch",
    "email_sendemail": "emailDraft",
    "general_greet": "chat",
    "general_joke": "chat",
    "general_quirky": "chat",
    "lists_createoradd": "reminder",
    "lists_query": "reminder",
    "news_query": "webSearch",
    "qa_currency": "webSearch",
    "qa_definition": "chat",
    "qa_factoid": "chat",
    "qa_maths": "chat",
    "qa_stock": "webSearch",
    "recommendation_events": "webSearch",
    "recommendation_locations": "maps",
    "recommendation_movies": "webSearch",
    "transport_query": "maps",
    "transport_traffic": "maps",
    "weather_query": "weather",
}

_MASSIVE_CORTEX_TOOL_BY_ROUTE: dict[tuple[str, str], str] = {
    ("alarm_query", "alarm"): "alarm.list",
    ("alarm_remove", "alarm"): "alarm.cancel",
    ("alarm_set", "alarm"): "alarm.schedule",
    ("calendar_query", "calendar"): "calendar.list",
    ("calendar_set", "calendar"): "calendar.create",
    ("calendar_set", "reminder"): "reminders.create",
    ("cooking_query", "webSearch"): "web.search",
    ("cooking_recipe", "webSearch"): "web.search",
    ("email_query", "outlook"): "outlook.messages.search",
    ("email_querycontact", "contactSearch"): "contacts.search",
    ("email_sendemail", "emailDraft"): "mail.draft",
    ("lists_createoradd", "reminder"): "reminders.create",
    ("lists_query", "reminder"): "reminders.list",
    ("news_query", "webSearch"): "web.search",
    ("qa_currency", "webSearch"): "web.search",
    ("qa_stock", "webSearch"): "web.search",
    ("recommendation_events", "webSearch"): "web.search",
    ("recommendation_locations", "maps"): "maps.search",
    ("recommendation_movies", "webSearch"): "web.search",
    ("transport_query", "maps"): "maps.directions",
    ("transport_query", "webSearch"): "web.search",
    ("transport_traffic", "maps"): "maps.search",
    ("weather_query", "weather"): "weather",
}


def _transform_massive(
    source: dict[str, Any], artifact: Path, policy_version: str, contract: LumenContract
) -> list[dict[str, Any]]:
    rows = list(_read_tar_jsonl_member(artifact, source["artifactMember"]))
    cortex_candidates: list[dict[str, Any]] = []
    fleet_candidates: list[dict[str, Any]] = []
    executor_candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("partition") != source["sourcePartition"] or row.get("locale") != source.get("locale"):
            continue
        utterance = _clean_massive_utterance(row.get("utt"))
        upstream_intent = row.get("intent")
        if utterance is None or not isinstance(upstream_intent, str):
            continue
        lumen_intent = _massive_lumen_intent(row, utterance)
        if lumen_intent is None or lumen_intent not in contract.intents:
            continue
        opaque_key = _opaque_group_hash(source["id"], str(row.get("id")), utterance)
        source_content_hash = _source_content_sha256(row)
        cortex_target = _massive_cortex_target(upstream_intent, lumen_intent, contract)
        if cortex_target is not None:
            selected_tool_id = cortex_target.get("selectedToolID")
            cortex = _make_record(
                source,
                policy_version,
                agent="cortex",
                user=utterance,
                assistant=_canonical_json(cortex_target),
                task_type="public_intent_routing",
                transformation="massive_intent_to_native_cortex_envelope",
                group_hash=opaque_key,
                source_content_sha256=source_content_hash,
                source_path=source["artifactMember"],
                stratum=upstream_intent,
                language="en",
                quality={"expertAnnotated": False, "humanAnnotated": True, "humanReviewed": False, "synthetic": False},
                tool_ids=[selected_tool_id] if isinstance(selected_tool_id, str) else [],
            )
            if cortex:
                cortex_candidates.append(cortex)
        fleet_target = _canonical_json(
            {
                "capability": lumen_intent,
                "delegatePlanningTo": "cortex",
                "delegateExecutionTo": "executor" if lumen_intent != "chat" else None,
                "userResponseOwner": "mouth",
            }
        )
        fleet = _make_record(
            source,
            policy_version,
            agent="fleet",
            user=f"Which Lumen capability owns this request?\n\n{utterance}",
            assistant=fleet_target,
            task_type="public_capability_delegation",
            transformation="massive_intent_to_fleet_boundary",
            group_hash=opaque_key,
            source_content_sha256=source_content_hash,
            source_path=source["artifactMember"],
            stratum=upstream_intent,
            language="en",
            quality={"expertAnnotated": False, "humanAnnotated": True, "humanReviewed": False, "synthetic": False},
        )
        if fleet:
            fleet_candidates.append(fleet)
        tool_call = _massive_lumen_tool_call(row)
        if tool_call is not None and _validate_tool_call(tool_call, contract.tools):
            executor = _make_record(
                source,
                policy_version,
                agent="executor",
                user=utterance,
                assistant=_canonical_json(tool_call),
                task_type="public_manifest_tool_argument_extraction",
                transformation="massive_slots_to_lumen_tool_envelope",
                group_hash=opaque_key,
                source_content_sha256=source_content_hash,
                source_path=source["artifactMember"],
                stratum=tool_call["tool"],
                language="en",
                quality={"expertAnnotated": False, "humanAnnotated": True, "humanReviewed": False, "synthetic": False},
                tool_ids=[tool_call["tool"]],
            )
            if executor:
                executor_candidates.append(executor)
    return [*cortex_candidates, *fleet_candidates, *executor_candidates]


def _massive_cortex_target(
    upstream_intent: str,
    lumen_intent: str,
    contract: LumenContract,
) -> dict[str, Any] | None:
    intent_contract = contract.intents.get(lumen_intent)
    if not isinstance(intent_contract, dict):
        return None
    allowed_tool_ids = intent_contract.get("allowedToolIDs")
    if not isinstance(allowed_tool_ids, list) or any(not isinstance(tool_id, str) for tool_id in allowed_tool_ids):
        return None

    if lumen_intent == "chat":
        target: dict[str, Any] = {
            "intent": lumen_intent,
            "selectedToolID": None,
            "requiresApproval": False,
            "nextModel": "mouth",
            "reasoningSummary": "The manifest routes this request as chat without a native tool action.",
        }
    else:
        selected_tool_id = _MASSIVE_CORTEX_TOOL_BY_ROUTE.get((upstream_intent, lumen_intent))
        tool = contract.tools.get(selected_tool_id) if selected_tool_id is not None else None
        if tool is None or selected_tool_id not in allowed_tool_ids:
            return None
        requires_approval = tool.get("requiresApproval") is True
        target = {
            "intent": lumen_intent,
            "selectedToolID": selected_tool_id,
            "requiresApproval": requires_approval,
            "nextModel": "approval" if requires_approval else "executor",
            "reasoningSummary": (
                f"The manifest allows {selected_tool_id} for {lumen_intent}; persist the tool action before finalization."
            ),
            "actionStep": {
                "type": "tool_call",
                "toolID": selected_tool_id,
                "mustPersistBeforeFinal": True,
            },
        }
    return target if _validate_cortex_target(target, contract) else None


def _validate_cortex_target(target: Any, contract: LumenContract) -> bool:
    if not isinstance(target, dict) or any(field not in target for field in contract.cortex_required_fields):
        return False
    intent = target.get("intent")
    selected_tool_id = target.get("selectedToolID")
    if not isinstance(intent, str) or intent not in contract.intents:
        return False
    if type(target.get("requiresApproval")) is not bool:
        return False
    if not isinstance(target.get("nextModel"), str) or not isinstance(target.get("reasoningSummary"), str):
        return False
    allowed_tool_ids = contract.intents[intent].get("allowedToolIDs")
    if not isinstance(allowed_tool_ids, list):
        return False
    if selected_tool_id is None:
        return (
            intent == "chat"
            and target["requiresApproval"] is False
            and target["nextModel"] == "mouth"
            and "actionStep" not in target
        )
    tool = contract.tools.get(selected_tool_id) if isinstance(selected_tool_id, str) else None
    action_step = target.get("actionStep")
    expected_requires_approval = tool.get("requiresApproval") is True if tool is not None else False
    return (
        tool is not None
        and selected_tool_id in allowed_tool_ids
        and target["requiresApproval"] is expected_requires_approval
        and target["nextModel"] == ("approval" if expected_requires_approval else "executor")
        and isinstance(action_step, dict)
        and action_step.get("type") == "tool_call"
        and action_step.get("toolID") == selected_tool_id
        and action_step.get("mustPersistBeforeFinal") is True
    )


def _massive_lumen_tool_call(row: Mapping[str, Any]) -> dict[str, Any] | None:
    intent = row.get("intent")
    utterance = _clean_massive_utterance(row.get("utt"))
    if utterance is None:
        return None
    if _massive_lumen_intent(row, utterance) is None:
        return None
    slots = _massive_slots(str(row.get("annot_utt") or ""))
    if intent == "alarm_query":
        return {"tool": "alarm.list", "arguments": {}}
    if intent == "calendar_query":
        return {"tool": "calendar.list", "arguments": {}}
    if intent == "lists_query":
        return {"tool": "reminders.list", "arguments": {}}
    if intent == "weather_query":
        locations = slots.get("place_name", [])
        return {"tool": "weather", "arguments": ({"city": locations[0]} if len(locations) == 1 else {})}
    if intent == "recommendation_locations":
        terms = [*slots.get("business_type", []), *slots.get("food_type", []), *slots.get("place_name", [])]
        query = " ".join(dict.fromkeys(terms)).strip() or utterance
        return {"tool": "maps.search", "arguments": {"query": query}}
    if intent == "email_querycontact":
        terms = [*slots.get("person", []), *slots.get("relation", []), *slots.get("business_name", [])]
        if terms:
            return {"tool": "contacts.search", "arguments": {"query": " ".join(dict.fromkeys(terms))}}
        return None
    if intent in {
        "cooking_query",
        "cooking_recipe",
        "news_query",
        "qa_currency",
        "qa_stock",
        "recommendation_events",
        "recommendation_movies",
    }:
        return {"tool": "web.search", "arguments": {"query": utterance}}
    return None


def _clean_massive_utterance(value: Any) -> str | None:
    utterance = _clean_text(value)
    if utterance is None:
        return None
    utterance = _WAKE_WORD_RE.sub("", utterance).strip()
    if len(utterance.split()) < 3:
        return None
    return utterance or None


def _massive_lumen_intent(row: Mapping[str, Any], utterance: str) -> str | None:
    upstream_intent = row.get("intent")
    if not isinstance(upstream_intent, str):
        return None
    lowered = utterance.lower()
    slots = _massive_slots(str(row.get("annot_utt") or ""))

    if upstream_intent in {"lists_createoradd", "lists_query"}:
        if re.search(r"\b(?:music|play\s*list|playlist|song|album|rap)\b", lowered):
            return None
        return "reminder"
    if upstream_intent == "calendar_set":
        if re.search(r"\b(?:memo|note|remind|reminder|to[ -]?do)\b", lowered):
            return "reminder"
        return "calendar"
    if upstream_intent == "email_querycontact":
        contact_terms = [*slots.get("person", []), *slots.get("relation", []), *slots.get("business_name", [])]
        return "contactSearch" if contact_terms else None
    if upstream_intent == "transport_query":
        if re.search(r"\b(?:arriv|bus|flight|platform|schedule|subway|ticket|train)\w*\b", lowered):
            return "webSearch"
        if re.search(r"\b(?:direction|drive|get to|map|route|travel time|walk)\w*\b", lowered):
            return "maps"
        return None
    return _MASSIVE_LUMEN_INTENTS.get(upstream_intent)


def _massive_slots(annotated: str) -> dict[str, list[str]]:
    slots: dict[str, list[str]] = defaultdict(list)
    for raw_name, raw_value in _SLOT_RE.findall(annotated):
        name = raw_name.strip()
        value = " ".join(raw_value.split())
        if name and value:
            slots[name].append(value)
    return dict(slots)


def _validate_tool_call(call: Any, tools: Mapping[str, dict[str, Any]]) -> bool:
    if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
        return False
    tool_id = call.get("tool")
    arguments = call.get("arguments")
    tool = tools.get(tool_id) if isinstance(tool_id, str) else None
    if tool is None or not isinstance(arguments, dict):
        return False
    definitions = tool.get("arguments")
    if not isinstance(definitions, list):
        return False
    by_name = {item.get("name"): item for item in definitions if isinstance(item, dict) and isinstance(item.get("name"), str)}
    if any(name not in by_name for name in arguments):
        return False
    if any(item.get("required") is True and name not in arguments for name, item in by_name.items()):
        return False
    for name, value in arguments.items():
        definition = by_name[name]
        if not _matches_manifest_type(value, str(definition.get("type") or "")):
            return False
        allowed = definition.get("allowedValues")
        if isinstance(allowed, list) and allowed and value not in allowed:
            return False
    return True


def _matches_manifest_type(value: Any, declared: str) -> bool:
    normalized = declared.lower()
    if normalized in {"string", "url", "date"}:
        return isinstance(value, str) and bool(value.strip())
    if normalized in {"number", "double", "float"}:
        return type(value) in {int, float}
    if normalized in {"int", "integer"}:
        return type(value) is int
    if normalized in {"bool", "boolean"}:
        return type(value) is bool
    if normalized in {"array", "list"}:
        return isinstance(value, list)
    if normalized in {"object", "dictionary", "dict"}:
        return isinstance(value, dict)
    if normalized in {"null", "nil", "none"}:
        return value is None
    return False


def _transform_oasst2(source: dict[str, Any], artifact: Path, policy_version: str) -> list[dict[str, Any]]:
    messages = list(_read_parquet_rows(artifact))
    by_id = {message.get("message_id"): message for message in messages if isinstance(message, dict)}
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        language = message.get("lang")
        if language not in source.get("languageCaps", {}):
            continue
        parent = by_id.get(message.get("parent_id"))
        if not _oasst_message_is_eligible(message, assistant=True):
            continue
        if not isinstance(parent, dict) or parent.get("role") != "prompter" or parent.get("lang") != language:
            continue
        if parent.get("parent_id") is not None or not _oasst_message_is_eligible(parent, assistant=False):
            continue
        user = _clean_text(parent.get("text"))
        assistant = _clean_text(message.get("text"))
        if user is None or assistant is None:
            continue
        if _has_disallowed_public_content(user, assistant):
            continue
        user_token_count = len(user.split())
        assistant_token_count = len(assistant.split())
        if not (3 <= user_token_count <= 256 and 18 <= assistant_token_count <= 180):
            continue
        tree_id = str(message.get("message_tree_id") or "")
        if not tree_id or tree_id != str(parent.get("message_tree_id") or ""):
            continue
        group_hash = _opaque_group_hash(source["id"], tree_id)
        source_content_hash = _source_content_sha256({"prompt": parent, "response": message})
        grounded_example = _grounded_oasst_example(user, assistant, str(language))
        if grounded_example is None:
            continue
        finalizer_prompt, final_answer = grounded_example
        record = _make_record(
            source,
            policy_version,
            agent="mouth",
            user=finalizer_prompt,
            assistant=final_answer,
            task_type="public_grounded_response_finalization",
            transformation="oasst2_source_observations_to_concise_final",
            group_hash=group_hash,
            source_content_sha256=source_content_hash,
            source_path=source["artifactPath"],
            stratum=str(language),
            language=str(language),
            quality={
                "expertAnnotated": False,
                "humanReviewed": False,
                "synthetic": False,
                "upstreamHumanReviewed": True,
                "upstreamRank": 0,
                "upstreamReviewCountMinimum": 3,
            },
        )
        if record:
            by_language[str(language)].append(record)
    language_caps = source.get("languageCaps", {})
    if not any(
        isinstance(language, str) and type(cap) is int and cap > 0
        for language, cap in language_caps.items()
    ):
        raise PublicCorpusError(f"Source {source['id']} requires positive languageCaps")
    # Keep the complete eligible pool here. The shared post-transform selector
    # scores every candidate before applying balanced language caps.
    return [
        record
        for language in sorted(by_language)
        for record in by_language[language]
    ]


def _read_parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise PublicCorpusError(
            "Building the OASST2 train source requires pyarrow; run with `uv run --with pyarrow`"
        ) from error
    parquet_file = parquet.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            if not isinstance(row, dict):
                raise PublicCorpusError("OASST2 parquet row must be an object")
            yield row


def _oasst_message_is_eligible(message: Mapping[str, Any], *, assistant: bool) -> bool:
    if message.get("deleted") is True or message.get("synthetic") is True:
        return False
    if message.get("review_result") is not True or message.get("tree_state") != "ready_for_export":
        return False
    review_count = message.get("review_count")
    if type(review_count) is not int or review_count < 3:
        return False
    if assistant and message.get("rank") != 0:
        return False
    text = message.get("text")
    if not isinstance(text, str) or _has_disallowed_public_content(text):
        return False
    labels = _oasst_labels(message.get("labels"))
    unsafe_labels = ["spam", "lang_mismatch", "pii", "not_appropriate", "hate_speech", "sexual_content", "toxicity", "violence"]
    if assistant:
        unsafe_labels.append("fails_task")
    for unsafe in unsafe_labels:
        if unsafe not in labels or labels[unsafe] > 0.05:
            return False
    if assistant and labels.get("quality", 0.0) < 0.75:
        return False
    if assistant and labels.get("helpfulness", 0.0) < 0.70:
        return False
    detoxify = message.get("detoxify")
    if assistant and not isinstance(detoxify, dict):
        return False
    if isinstance(detoxify, dict):
        for key in ("toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit"):
            value = detoxify.get(key)
            if type(value) not in {int, float} or value > 0.05:
                return False
    return True


def _has_disallowed_public_content(*values: str) -> bool:
    patterns = (
        _CODE_RE,
        _HIGH_STAKES_RE,
        _CURRENT_RECOMMENDATION_RE,
        _SECURITY_RE,
        _MODEL_IDENTITY_RE,
        _COPYRIGHT_STYLE_RE,
        _UNSAFE_EDIT_RE,
        _CORPUS_ARTIFACT_RE,
    )
    return any(pattern.search(value) for value in values for pattern in patterns)


def _grounded_oasst_example(user: str, source_answer: str, language: str) -> tuple[str, str] | None:
    """Convert a safe upstream answer into an observation-to-final example.

    The model target is a deterministic, shorter selection from explicit source
    observations. The upstream answer is never labelled as verified or trusted,
    and the target is never a byte-identical copy of that answer.
    """

    claims = _extract_grounded_claims(source_answer)
    if len(claims) < 3:
        return None
    observations = claims[:3]
    selected = observations[:2]
    if any(not _grounded_claim_is_answer(claim) for claim in selected):
        return None
    if sum(len(claim.split()) for claim in selected) > 64:
        return None
    if language == "fr":
        prompt = (
            "Rédigez une réponse finale concise en utilisant uniquement les observations 1 et 2. "
            "N'ajoutez aucun fait.\n\n"
            f"Demande de l'utilisateur :\n{user}\n\n"
            "Observations sources :\n"
            + "\n".join(f"{index}. {claim}" for index, claim in enumerate(observations, start=1))
        )
    else:
        prompt = (
            "Write a concise final answer using only source observations 1 and 2. Do not add facts.\n\n"
            f"User request:\n{user}\n\n"
            "Source observations:\n"
            + "\n".join(f"{index}. {claim}" for index, claim in enumerate(observations, start=1))
        )
    final_answer = "\n".join(f"- {claim}" for claim in selected)
    if final_answer.strip() == source_answer.strip() or len(final_answer) >= len(source_answer):
        return None
    if _has_disallowed_public_content(prompt, final_answer) or not _well_formed_output(final_answer):
        return None
    return prompt, final_answer


def _extract_grounded_claims(value: str) -> list[str]:
    lines: list[str] = []
    for raw_line in value.replace("\r", "\n").split("\n"):
        line = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", raw_line).strip()
        if not line or _META_RESPONSE_RE.search(line) or line.endswith(":"):
            continue
        lines.extend(re.split(r"(?<=[.!?])\s+", line))
    claims: list[str] = []
    seen: set[str] = set()
    for candidate in lines:
        claim = " ".join(candidate.split()).strip(" -*•")
        words = claim.split()
        if (
            not (4 <= len(words) <= 32)
            or _META_RESPONSE_RE.search(claim)
            or not _grounded_claim_is_answer(claim)
        ):
            continue
        if not re.search(r"[.!?]$", claim):
            claim += "."
        key = claim.casefold()
        if key in seen or not _well_formed_output(claim):
            continue
        seen.add(key)
        claims.append(claim)
    return claims


def _grounded_claim_is_answer(value: str) -> bool:
    """Return true only for source observations that can support a final answer.

    OASST root responses sometimes answer an underspecified prompt with several
    clarification questions. Those turns are useful conversational data, but a
    question or refusal is not a trusted observation and must not be converted
    into a factual finalizer target.
    """

    stripped = value.strip()
    if not stripped or "?" in stripped:
        return False
    if _QUESTION_LEAD_RE.search(stripped) or _QUESTION_FRAGMENT_RE.search(stripped):
        return False
    if _NON_ANSWER_RE.search(stripped) or _META_RESPONSE_RE.search(stripped):
        return False
    return True


def _oasst_labels(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or not isinstance(value.get("name"), list) or not isinstance(value.get("value"), list):
        return {}
    out: dict[str, float] = {}
    for name, score in zip(value["name"], value["value"]):
        if isinstance(name, str) and type(score) in {int, float}:
            out[name] = float(score)
    return out


def _transform_coedit(source: dict[str, Any], artifact: Path, policy_version: str) -> list[dict[str, Any]]:
    mimicry_candidates: list[dict[str, Any]] = []
    rem_candidates: list[dict[str, Any]] = []
    for row in _read_jsonl(artifact.open("rt", encoding="utf-8")):
        task = row.get("task")
        source_text = _clean_text(row.get("src"))
        target_text = _clean_text(row.get("tgt"))
        if source_text is None or target_text is None or source_text == target_text:
            continue
        if _has_disallowed_public_content(source_text, target_text):
            continue
        editable_text = _coedit_editable_text(source_text)
        if not _meaning_preserving_edit(editable_text, target_text):
            continue
        group_hash = _opaque_group_hash(source["id"], str(row.get("_id")), source_text, target_text)
        source_content_hash = _source_content_sha256(row)
        if task in {"clarity", "coherence", "neutralize", "simplification"}:
            record = _make_record(
                source,
                policy_version,
                agent="mimicry",
                user=source_text,
                assistant=target_text,
                task_type="public_meaning_preserving_style_edit",
                transformation="coedit_meaning_preserving_edit",
                group_hash=group_hash,
                source_content_sha256=source_content_hash,
                source_path=source["artifactPath"],
                stratum=str(task),
                language="en",
                quality={
                    "expertAnnotated": False,
                    "humanReviewed": False,
                    "synthetic": False,
                    "sourceTask": task,
                    "upstreamCurated": True,
                },
            )
            if record:
                mimicry_candidates.append(record)
        elif task == "gec":
            repair_prompt = (
                "Diagnose and repair this text while preserving its meaning, entities, numbers, and negation. "
                "Return the diagnosis and repaired text as JSON.\n\n"
                f"Candidate:\n{editable_text}"
            )
            record = _make_record(
                source,
                policy_version,
                agent="rem",
                user=repair_prompt,
                assistant=_canonical_json(
                    {
                        "diagnosis": "grammar_or_usage_error",
                        "preserveMeaning": True,
                        "repair": target_text,
                    }
                ),
                task_type="public_text_repair",
                transformation="coedit_grammatical_error_repair",
                group_hash=group_hash,
                source_content_sha256=source_content_hash,
                source_path=source["artifactPath"],
                stratum="gec",
                language="en",
                quality={
                    "expertAnnotated": False,
                    "humanReviewed": False,
                    "synthetic": False,
                    "sourceTask": task,
                    "upstreamCurated": True,
                },
            )
            if record:
                rem_candidates.append(record)
    return [*mimicry_candidates, *rem_candidates]


def _coedit_editable_text(source: str) -> str:
    stripped = _INSTRUCTION_PREFIX_RE.sub("", source, count=1).strip()
    if stripped != source:
        return stripped
    prefix, separator, remainder = source.partition(":")
    if (
        separator
        and len(prefix.split()) <= 14
        and re.search(r"(?i)\b(?:clarif|clear|coher|cohes|neutral|point of view|simpl|grammar|grammat)", prefix)
    ):
        return remainder.strip()
    return source


def _meaning_preserving_edit(source: str, target: str) -> bool:
    if len(source) < 8 or len(target) < 8:
        return False
    if len(source) > MAX_TEXT_CHARS or len(target) > MAX_TEXT_CHARS:
        return False
    source_sequence = re.findall(r"(?u)\b\w+(?:['’]\w+)?\b", source.lower())
    target_sequence = re.findall(r"(?u)\b\w+(?:['’]\w+)?\b", target.lower())
    if not source_sequence or not target_sequence or len(target_sequence) > 160 or len(source_sequence) > 220:
        return False
    if _has_disallowed_public_content(source, target):
        return False
    if not _well_formed_output(source) or not _well_formed_output(target):
        return False
    if _semantic_number_counter(source) != _semantic_number_counter(target):
        return False
    if Counter(_URL_RE.findall(source)) != Counter(_URL_RE.findall(target)):
        return False
    if Counter(match.group(0).lower() for match in _NEGATION_RE.finditer(source)) != Counter(
        match.group(0).lower() for match in _NEGATION_RE.finditer(target)
    ):
        return False
    if Counter(match.group(0).casefold() for match in _TEMPORAL_FACT_RE.finditer(source)) != Counter(
        match.group(0).casefold() for match in _TEMPORAL_FACT_RE.finditer(target)
    ):
        return False
    if _capitalized_entity_counter(source) != _capitalized_entity_counter(target):
        return False
    if Counter(match.group(0).casefold() for match in _MEASUREMENT_UNIT_RE.finditer(source)) != Counter(
        match.group(0).casefold() for match in _MEASUREMENT_UNIT_RE.finditer(target)
    ):
        return False
    if _factual_status_anchor_set(source) != _factual_status_anchor_set(target):
        return False
    source_tokens = set(source_sequence)
    target_tokens = set(target_sequence)
    overlap = len(source_tokens & target_tokens) / min(len(source_tokens), len(target_tokens))
    length_ratio = min(len(source), len(target)) / max(len(source), len(target))
    edit_ratio = 1.0 - SequenceMatcher(a=source_sequence, b=target_sequence, autojunk=False).ratio()
    return overlap >= 0.75 and length_ratio >= 0.75 and 0.03 <= edit_ratio <= 0.35


def _factual_status_anchor_set(value: str) -> frozenset[str]:
    """Normalize epistemic qualifiers without treating a rewrite as fact creation.

    Surface synonyms such as ``asserted`` and ``indicated`` share an
    attribution anchor. Removing that anchor entirely, or adding a qualifier
    such as ``may`` or ``confirmed``, changes factual status and fails closed.
    """

    return frozenset(name for name, pattern in _FACTUAL_STATUS_PATTERNS if pattern.search(value))


def _semantic_number_counter(value: str) -> Counter[str]:
    tokens = [match.group(0).casefold() for match in _NUMBER_RE.finditer(value)]
    tokens.extend(match.group(0).casefold() for match in _DIGIT_ORDINAL_RE.finditer(value))
    tokens.extend(match.group(0).casefold() for match in _NUMBER_WORD_RE.finditer(value))
    return Counter(tokens)


def _capitalized_entity_counter(value: str) -> Counter[str]:
    tokens = re.findall(r"(?<![\w'’])(?:[A-Z]{2,}|[A-Z][A-Za-z0-9'’.-]{1,})(?![\w'’])", value)
    return Counter(token.casefold() for token in tokens)


def _well_formed_output(value: str) -> bool:
    stripped = value.strip()
    if not stripped or _CORPUS_ARTIFACT_RE.search(stripped) or _MALFORMED_TEXT_RE.search(stripped):
        return False
    if _DANGLING_END_RE.search(stripped.rstrip(".?!\"')]}")):
        return False
    if stripped[-1] not in ".?!\"')]}":
        return False
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    if any(stripped.count(opening) != stripped.count(closing) for opening, closing in pairs):
        return False
    if stripped.count('"') % 2 != 0:
        return False
    if re.search(r"(?i)\b([a-z]+)\s+\1\b", stripped):
        return False
    return True


def _quality_balanced_coedit_select(
    records: Sequence[dict[str, Any]], cap: int
) -> list[dict[str, Any]]:
    if cap <= 0:
        return []
    deduplicated: dict[str, dict[str, Any]] = {}
    for record in records:
        deduplicated.setdefault(record["metadata"]["publicCorpus"]["transformedContentSHA256"], record)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in deduplicated.values():
        strata[record["metadata"]["publicCorpus"]["stratum"]].append(record)

    def quality_key(record: dict[str, Any]) -> tuple[float, float, str]:
        source = _coedit_editable_text(record["messages"][0]["content"])
        target = record["messages"][1]["content"]
        source_tokens = re.findall(r"(?u)\b\w+(?:['’]\w+)?\b", source.casefold())
        target_tokens = re.findall(r"(?u)\b\w+(?:['’]\w+)?\b", target.casefold())
        similarity = SequenceMatcher(a=source_tokens, b=target_tokens, autojunk=False).ratio()
        length_ratio = min(len(source), len(target)) / max(len(source), len(target))
        return (-similarity, -length_ratio, record["id"])

    for items in strata.values():
        items.sort(key=quality_key)
    selected: list[dict[str, Any]] = []
    names = sorted(strata)
    index = 0
    while len(selected) < cap:
        progressed = False
        for name in names:
            items = strata[name]
            if index < len(items):
                selected.append(items[index])
                progressed = True
                if len(selected) == cap:
                    break
        if not progressed:
            break
        index += 1
    return selected


def _transform_json_schema_tests(source: dict[str, Any], artifact: Path, policy_version: str) -> list[dict[str, Any]]:
    selected_files = set(source.get("selectedFiles", []))
    prefix = str(source["artifactMemberPrefix"])
    candidates: list[dict[str, Any]] = []
    with tarfile.open(artifact, "r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            relative = _tar_relative_member(member.name, prefix)
            if relative is None or relative not in selected_files or not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            groups = json.load(handle)
            if not isinstance(groups, list):
                raise PublicCorpusError(f"JSON Schema test file must contain a list: {relative}")
            for group_index, group in enumerate(groups):
                if not isinstance(group, dict) or "schema" not in group or not isinstance(group.get("tests"), list):
                    continue
                schema = group["schema"]
                group_description = str(group.get("description") or relative)
                valid_examples = [
                    test.get("data")
                    for test in group["tests"]
                    if isinstance(test, dict) and test.get("valid") is True and "data" in test
                ]
                valid_example = valid_examples[0] if valid_examples else None
                for test_index, test in enumerate(group["tests"]):
                    if not isinstance(test, dict) or test.get("valid") is not False or "data" not in test:
                        continue
                    case_description = str(test.get("description") or "invalid instance")
                    user = (
                        "A strict JSON boundary rejected this instance. Diagnose the validation invariant and return JSON only.\n"
                        f"Schema: {_canonical_json(schema)}\nInstance: {_canonical_json(test['data'])}"
                    )
                    assistant = _canonical_json(
                        {
                            "decision": "reject",
                            "valid": False,
                            "invariant": group_description,
                            "case": case_description,
                            "repair": {
                                "action": "revise the instance and revalidate before execution",
                                "knownValidExample": valid_example,
                            },
                        }
                    )
                    group_hash = _opaque_group_hash(source["id"], relative, str(group_index), str(test_index))
                    source_content_hash = _source_content_sha256(
                        {
                            "file": relative,
                            "groupDescription": group_description,
                            "schema": schema,
                            "test": test,
                        }
                    )
                    record = _make_record(
                        source,
                        policy_version,
                        agent="rem",
                        user=user,
                        assistant=assistant,
                        task_type="public_schema_failure_diagnosis",
                        transformation="json_schema_invalid_case_to_fail_closed_repair",
                        group_hash=group_hash,
                        source_content_sha256=source_content_hash,
                        source_path=f"{prefix}{relative}",
                        stratum=relative,
                        language="en",
                        quality={"expertAnnotated": True, "humanReviewed": True, "synthetic": False, "draft": "2020-12"},
                    )
                    if record:
                        candidates.append(record)
    return candidates


def _transform_apigen_xlam(
    source: dict[str, Any], artifact: Path, policy_version: str, contract: LumenContract
) -> list[dict[str, Any]]:
    _validate_apigen_xlam_source_contract(source)
    rows = _read_apigen_xlam_rows(source, artifact)
    mappings = source["toolMappings"]
    candidates: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        query = _apigen_xlam_query(row)
        definitions = _apigen_xlam_tool_definitions(row.get("tools", row.get("functions")))
        calls = _apigen_xlam_calls(row)
        if query is None or definitions is None or len(calls) != 1:
            continue
        upstream_name, upstream_arguments = calls[0]
        mapping = mappings.get(upstream_name)
        definition = definitions.get(upstream_name)
        if not isinstance(mapping, dict) or definition is None:
            continue
        argument_mappings = mapping["argumentMappings"]
        if set(upstream_arguments) != set(argument_mappings):
            continue
        if not _apigen_xlam_definition_accepts_call(definition, upstream_arguments):
            continue
        lumen_call = {
            "tool": mapping["toolID"],
            "arguments": {
                argument_mappings[source_name]: value
                for source_name, value in upstream_arguments.items()
            },
        }
        serialized_call = _canonical_json(lumen_call)
        if (
            not _validate_tool_call(lumen_call, contract.tools)
            or _has_disallowed_toolace_content(query, serialized_call)
        ):
            continue

        group_hash = _opaque_group_hash(source["id"], str(row_index), upstream_name)
        source_content_hash = _source_content_sha256(
            {"arguments": upstream_arguments, "query": query, "tool": upstream_name}
        )
        shared_quality = {
            "expertAnnotated": False,
            "humanReviewed": False,
            "synthetic": True,
            "upstreamFormatVerified": True,
            "lumenManifestValidated": True,
            "exactToolMapping": True,
        }
        if "executor" in source["targetAdapters"]:
            executor = _make_record(
                source,
                policy_version,
                agent="executor",
                user=query,
                assistant=serialized_call,
                task_type="public_verified_tool_argument_extraction",
                transformation="apigen_xlam_call_to_manifest_exact_lumen_envelope",
                group_hash=group_hash,
                source_content_sha256=source_content_hash,
                source_path=source["artifactPath"],
                stratum=mapping["toolID"],
                language="en",
                quality=shared_quality,
                tool_ids=[mapping["toolID"]],
            )
            if executor is not None:
                candidates.append(executor)
        if "cortex" in source["targetAdapters"]:
            cortex_target = _toolace_cortex_target(mapping["intent"], mapping["toolID"], contract)
            if cortex_target is None:
                continue
            cortex = _make_record(
                source,
                policy_version,
                agent="cortex",
                user=query,
                assistant=_canonical_json(cortex_target),
                task_type="public_verified_tool_routing",
                transformation="apigen_xlam_call_to_native_cortex_envelope",
                group_hash=group_hash,
                source_content_sha256=source_content_hash,
                source_path=source["artifactPath"],
                stratum=str(mapping["intent"]),
                language="en",
                quality=shared_quality,
                tool_ids=[mapping["toolID"]],
            )
            if cortex is not None:
                candidates.append(cortex)
    return candidates


def _read_apigen_xlam_rows(source: Mapping[str, Any], artifact: Path) -> list[dict[str, Any]]:
    if source["artifactFormat"] == "jsonl":
        try:
            return list(_read_jsonl(artifact.open("rt", encoding="utf-8")))
        except OSError as error:
            raise PublicCorpusError(f"Unable to read APIGen/xLAM JSONL artifact: {error}") from error
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicCorpusError(f"Unable to read APIGen/xLAM JSON artifact: {error}") from error
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        containers = [payload.get(key) for key in ("data", "rows") if payload.get(key) is not None]
        if len(containers) == 1 and isinstance(containers[0], list):
            return [row for row in containers[0] if isinstance(row, dict)]
    raise PublicCorpusError("APIGen/xLAM JSON artifact must be an array or contain one data/rows array")


def _apigen_xlam_query(row: Mapping[str, Any]) -> str | None:
    candidates = {
        cleaned
        for key in ("query", "prompt", "user")
        if (cleaned := _clean_text(row.get(key))) is not None
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _apigen_xlam_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _apigen_xlam_tool_definitions(value: Any) -> dict[str, dict[str, Any]] | None:
    decoded = _apigen_xlam_json_value(value)
    if isinstance(decoded, dict):
        decoded = [decoded]
    if not isinstance(decoded, list) or not decoded:
        return None
    definitions: dict[str, dict[str, Any]] = {}
    for item in decoded:
        if not isinstance(item, dict):
            return None
        definition = item.get("function") if item.get("type") == "function" else item
        if not isinstance(definition, dict):
            return None
        name = definition.get("name")
        if not isinstance(name, str) or not name.strip() or name in definitions:
            return None
        definitions[name] = definition
    return definitions


def _apigen_xlam_calls(row: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    parsed_candidates: list[list[tuple[str, dict[str, Any]]]] = []
    for key in ("answers", "calls", "tool_calls"):
        if key not in row or row[key] is None:
            continue
        decoded = _apigen_xlam_json_value(row[key])
        items = decoded if isinstance(decoded, list) else [decoded]
        parsed: list[tuple[str, dict[str, Any]]] = []
        for item in items:
            if not isinstance(item, dict):
                parsed = []
                break
            call = item.get("function") if isinstance(item.get("function"), dict) else item
            name = call.get("name") if isinstance(call, dict) else None
            arguments = _apigen_xlam_json_value(call.get("arguments")) if isinstance(call, dict) else None
            if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
                parsed = []
                break
            parsed.append((name, arguments))
        if not parsed:
            return []
        parsed_candidates.append(parsed)
    if not parsed_candidates:
        return []
    canonical = {_canonical_json(candidate) for candidate in parsed_candidates}
    return parsed_candidates[0] if len(canonical) == 1 else []


def _apigen_xlam_definition_accepts_call(
    definition: Mapping[str, Any], arguments: Mapping[str, Any]
) -> bool:
    parameters = definition.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type", "object") != "object":
        return False
    properties = parameters.get("properties")
    required = parameters.get("required", [])
    if (
        not isinstance(properties, dict)
        or not isinstance(required, list)
        or any(not isinstance(name, str) for name in required)
    ):
        return False
    return set(arguments) <= set(properties) and set(required) <= set(arguments)


_TOOLACE_CALL_MAPPINGS: dict[str, tuple[str, str, dict[str, str]]] = {
    "Get Weather Forecast by Location": ("weather", "weather", {"location": "location"}),
    "GetCurrentWeather": ("weather", "weather", {"city": "city"}),
    "GetWeatherForecast": ("weather", "weather", {"location": "location"}),
    "Realtime Weather API": ("weather", "weather", {"q": "city"}),
    "Weather By City Name": ("weather", "weather", {"q": "city"}),
    "Weather Data API": ("weather", "weather", {"location": "location"}),
    "Weather Forecast": ("weather", "weather", {"location": "location"}),
    "Search Nearby": ("maps.search", "maps", {"query": "query"}),
    "Search Contacts": ("contacts.search", "contactSearch", {"q": "query"}),
    "Web Search": ("web.search", "webSearch", {"q": "query"}),
    "read_text_file": ("files.read", "files", {"file_path": "name"}),
    "save_note": ("memory.save", "note", {"note": "content"}),
    "storeMemory": ("memory.save", "memory", {"knowledge": "content"}),
}


def _transform_toolace(
    source: dict[str, Any], artifact: Path, policy_version: str, contract: LumenContract
) -> list[dict[str, Any]]:
    try:
        rows = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicCorpusError(f"Unable to read ToolACE JSON artifact: {error}") from error
    if not isinstance(rows, list):
        raise PublicCorpusError("ToolACE JSON artifact must be an array")

    candidates: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        conversations = row.get("conversations") if isinstance(row, dict) else None
        if not isinstance(conversations, list):
            continue
        for turn_index, message in enumerate(conversations):
            if turn_index == 0 or not isinstance(message, dict) or message.get("from") != "assistant":
                continue
            previous = conversations[turn_index - 1]
            if not isinstance(previous, dict) or previous.get("from") != "user":
                continue
            user = _clean_text(previous.get("value"))
            source_call = _clean_text(message.get("value"))
            if (
                user is None
                or source_call is None
                or _ROLE_PROMPT_ARTIFACT_RE.search(user)
                or _has_disallowed_toolace_content(user, source_call)
            ):
                continue
            parsed_calls = _parse_toolace_calls(source_call)
            converted = [
                converted_call
                for name, arguments in parsed_calls
                if (converted_call := _toolace_lumen_call(name, arguments, contract)) is not None
            ]
            if not converted or len(converted) != len(parsed_calls):
                continue

            for call_index, (lumen_call, lumen_intent) in enumerate(converted):
                call_user = user
                if len(converted) > 1:
                    if lumen_call["tool"] != "files.read":
                        continue
                    call_user = f"Read the file named {lumen_call['arguments']['name']}."
                group_hash = _opaque_group_hash(
                    source["id"], str(row_index), str(turn_index), str(call_index)
                )
                source_content_hash = _source_content_sha256(
                    {"sourceCall": source_call, "user": user, "turn": turn_index}
                )
                shared_quality = {
                    "expertAnnotated": False,
                    "humanReviewed": False,
                    "synthetic": True,
                    "upstreamFormatVerified": True,
                    "lumenManifestValidated": True,
                }
                executor = _make_record(
                    source,
                    policy_version,
                    agent="executor",
                    user=call_user,
                    assistant=_canonical_json(lumen_call),
                    task_type="public_verified_tool_argument_extraction",
                    transformation="toolace_call_to_manifest_exact_lumen_envelope",
                    group_hash=group_hash,
                    source_content_sha256=source_content_hash,
                    source_path=source["artifactPath"],
                    stratum=lumen_call["tool"],
                    language="en",
                    quality=shared_quality,
                    tool_ids=[lumen_call["tool"]],
                )
                if executor is not None:
                    candidates.append(executor)

                cortex_target = _toolace_cortex_target(lumen_intent, lumen_call["tool"], contract)
                if cortex_target is not None:
                    cortex = _make_record(
                        source,
                        policy_version,
                        agent="cortex",
                        user=call_user,
                        assistant=_canonical_json(cortex_target),
                        task_type="public_verified_tool_routing",
                        transformation="toolace_call_to_native_cortex_envelope",
                        group_hash=group_hash,
                        source_content_sha256=source_content_hash,
                        source_path=source["artifactPath"],
                        stratum=lumen_intent,
                        language="en",
                        quality=shared_quality,
                        tool_ids=[lumen_call["tool"]],
                    )
                    if cortex is not None:
                        candidates.append(cortex)

                missing_argument = _first_required_argument(lumen_call["tool"], contract)
                if missing_argument is not None and missing_argument in lumen_call["arguments"]:
                    invalid_call = {
                        "tool": lumen_call["tool"],
                        "arguments": {
                            key: value
                            for key, value in lumen_call["arguments"].items()
                            if key != missing_argument
                        },
                    }
                    rem = _make_record(
                        source,
                        policy_version,
                        agent="rem",
                        user=(
                            "A structured tool call failed strict validation. Diagnose it and return JSON only.\n"
                            f"Request: {call_user}\nCandidate: {_canonical_json(invalid_call)}"
                        ),
                        assistant=_canonical_json(
                            {
                                "decision": "reject",
                                "failure": "missing_required_argument",
                                "missingArgument": missing_argument,
                                "tool": lumen_call["tool"],
                                "repair": "regenerate arguments against the current manifest before execution",
                            }
                        ),
                        task_type="public_tool_boundary_repair",
                        transformation="toolace_valid_call_to_missing_argument_repair",
                        group_hash=group_hash,
                        source_content_sha256=source_content_hash,
                        source_path=source["artifactPath"],
                        stratum=lumen_call["tool"],
                        language="en",
                        quality={**shared_quality, "derivedNegative": True},
                        tool_ids=[lumen_call["tool"]],
                    )
                    if rem is not None:
                        candidates.append(rem)

            if len(converted) != 1 or turn_index + 2 >= len(conversations):
                continue
            observation_message = conversations[turn_index + 1]
            final_message = conversations[turn_index + 2]
            if (
                not isinstance(observation_message, dict)
                or observation_message.get("from") != "tool"
                or not isinstance(final_message, dict)
                or final_message.get("from") != "assistant"
            ):
                continue
            observation = _clean_text(observation_message.get("value"))
            final = _clean_text(final_message.get("value"))
            if not _toolace_final_is_grounded(observation, final):
                continue
            lumen_call, _ = converted[0]
            mouth = _make_record(
                source,
                policy_version,
                agent="mouth",
                user=f"Trusted tool observation:\n{observation}\n\nUser request:\n{user}",
                assistant=final,
                task_type="public_observation_grounded_final",
                transformation="toolace_observation_to_grounded_final",
                group_hash=_opaque_group_hash(source["id"], str(row_index), str(turn_index)),
                source_content_sha256=_source_content_sha256(
                    {"final": final, "observation": observation, "sourceCall": source_call, "user": user}
                ),
                source_path=source["artifactPath"],
                stratum=lumen_call["tool"],
                language="en",
                quality={
                    "expertAnnotated": False,
                    "humanReviewed": False,
                    "synthetic": True,
                    "trustedObservationPresent": True,
                    "semanticAnchorChecked": True,
                },
                tool_ids=[lumen_call["tool"]],
            )
            if mouth is not None:
                candidates.append(mouth)
    return candidates


def _parse_toolace_calls(value: str) -> list[tuple[str, dict[str, Any]]]:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return []
    body = value[1:-1].strip()
    if not body:
        return []
    segments: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(body):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}" and depth > 0:
            depth -= 1
        elif character == "," and depth == 0:
            segments.append(body[start:index].strip())
            start = index + 1
    if quote is not None or depth != 0:
        return []
    segments.append(body[start:].strip())

    calls: list[tuple[str, dict[str, Any]]] = []
    for segment in segments:
        match = re.fullmatch(r"(.+?)\((.*)\)", segment, flags=re.DOTALL)
        if match is None:
            return []
        name = match.group(1).strip()
        try:
            parsed = ast.parse(f"f({match.group(2)})", mode="eval").body
        except SyntaxError:
            return []
        if not isinstance(parsed, ast.Call) or parsed.args:
            return []
        arguments: dict[str, Any] = {}
        for keyword in parsed.keywords:
            if keyword.arg is None or keyword.arg in arguments:
                return []
            try:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                return []
        calls.append((name, arguments))
    return calls


def _has_disallowed_toolace_content(*values: str) -> bool:
    return any(
        _contains_pii(value)
        or _CONTROL_RE.search(value)
        or _HIGH_STAKES_RE.search(value)
        or _SECURITY_RE.search(value)
        or _MODEL_IDENTITY_RE.search(value)
        or _COPYRIGHT_STYLE_RE.search(value)
        or _UNSAFE_EDIT_RE.search(value)
        for value in values
    )


def _toolace_lumen_call(
    name: str, arguments: Mapping[str, Any], contract: LumenContract
) -> tuple[dict[str, Any], str] | None:
    mapping = _TOOLACE_CALL_MAPPINGS.get(name)
    if mapping is None:
        return None
    tool_id, intent, argument_mapping = mapping
    mapped: dict[str, Any] = {}
    if name == "Search Nearby":
        if set(arguments) != {"query", "lng", "lat"}:
            return None
        query = arguments.get("query")
        longitude = arguments.get("lng")
        latitude = arguments.get("lat")
        if (
            not isinstance(query, str)
            or type(longitude) not in {int, float}
            or type(latitude) not in {int, float}
        ):
            return None
        mapped["query"] = f"{query} near latitude {latitude}, longitude {longitude}"
    elif set(arguments) != set(argument_mapping):
        return None
    for source_argument, target_argument in argument_mapping.items():
        if name == "Search Nearby":
            continue
        value = arguments.get(source_argument)
        if not isinstance(value, str) or _clean_text(value) is None:
            return None
        mapped[target_argument] = value
    if name in {"save_note", "storeMemory"}:
        mapped["kind"] = "note" if intent == "note" else "fact"
    call = {"tool": tool_id, "arguments": mapped}
    return (call, intent) if _validate_tool_call(call, contract.tools) else None


def _toolace_cortex_target(
    intent: str, tool_id: str, contract: LumenContract
) -> dict[str, Any] | None:
    intent_contract = contract.intents.get(intent)
    tool = contract.tools.get(tool_id)
    if (
        not isinstance(intent_contract, dict)
        or not isinstance(tool, dict)
        or tool_id not in (intent_contract.get("allowedToolIDs") or [])
    ):
        return None
    requires_approval = tool.get("requiresApproval") is True
    target = {
        "intent": intent,
        "selectedToolID": tool_id,
        "requiresApproval": requires_approval,
        "nextModel": "approval" if requires_approval else "executor",
        "reasoningSummary": f"The manifest allows {tool_id} for {intent}; persist the action before finalization.",
        "actionStep": {
            "type": "tool_call",
            "toolID": tool_id,
            "mustPersistBeforeFinal": True,
        },
    }
    return target if _validate_cortex_target(target, contract) else None


def _first_required_argument(tool_id: str, contract: LumenContract) -> str | None:
    tool = contract.tools.get(tool_id)
    arguments = tool.get("arguments") if isinstance(tool, dict) else None
    if not isinstance(arguments, list):
        return None
    required = sorted(
        argument["name"]
        for argument in arguments
        if isinstance(argument, dict)
        and isinstance(argument.get("name"), str)
        and argument.get("required") is not False
    )
    return required[0] if required else None


def _toolace_final_is_grounded(observation: str | None, final: str | None) -> bool:
    if (
        observation is None
        or final is None
        or _has_disallowed_toolace_content(observation, final)
        or _ROLE_PROMPT_ARTIFACT_RE.search(final)
    ):
        return False
    if _semantic_number_counter(final) - _semantic_number_counter(observation):
        return False
    observation_tokens = {token.lower() for token in _WORD_RE.findall(observation) if len(token) >= 4}
    final_tokens = {token.lower() for token in _WORD_RE.findall(final) if len(token) >= 4}
    if len(observation_tokens) < 3 or len(final_tokens) < 3:
        return False
    return len(observation_tokens & final_tokens) / len(final_tokens) >= 0.55


def _transform_faithdial(
    source: dict[str, Any], artifact: Path, policy_version: str
) -> list[dict[str, Any]]:
    try:
        dialogues = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicCorpusError(f"Unable to read FaithDial JSON artifact: {error}") from error
    if not isinstance(dialogues, list):
        raise PublicCorpusError("FaithDial JSON artifact must be an array")
    candidates: list[dict[str, Any]] = []
    for dialogue_index, dialogue in enumerate(dialogues):
        utterances = dialogue.get("utterances") if isinstance(dialogue, dict) else None
        if not isinstance(utterances, list):
            continue
        for utterance_index, utterance in enumerate(utterances):
            if not isinstance(utterance, dict) or utterance.get("speaker") != "Wizard":
                continue
            history = utterance.get("history")
            labels = utterance.get("BEGIN")
            if not isinstance(history, list) or not history or "Hallucination" not in (labels or []):
                continue
            request = _clean_text(history[-1])
            knowledge = _clean_text(utterance.get("knowledge"))
            chosen = _clean_text(utterance.get("response"))
            rejected = _clean_text(utterance.get("original_response"))
            if (
                request is None
                or knowledge is None
                or chosen is None
                or rejected is None
                or chosen == rejected
                or _has_disallowed_public_content(request, knowledge, chosen, rejected)
                or not _faithdial_response_is_supported(knowledge, chosen)
            ):
                continue
            prompt = f"Trusted observation:\n{knowledge}\n\nUser request:\n{request}"
            record = _make_record(
                source,
                policy_version,
                agent="mouth",
                user=prompt,
                assistant=chosen,
                task_type="public_grounded_response_preference",
                transformation="faithdial_knowledge_to_grounded_chosen_rejected",
                group_hash=_opaque_group_hash(source["id"], str(dialogue_index), str(utterance_index)),
                source_content_sha256=_source_content_sha256(
                    {
                        "chosen": chosen,
                        "knowledge": knowledge,
                        "rejected": rejected,
                        "request": request,
                    }
                ),
                source_path=source["artifactPath"],
                stratum="hallucination_correction",
                language="en",
                quality={
                    "expertAnnotated": False,
                    "humanReviewed": True,
                    "synthetic": False,
                    "trustedObservationPresent": True,
                    "humanCorrectedFinal": True,
                    "rejectedOriginalLabel": "Hallucination",
                },
                preference={"chosen": chosen, "rejected": rejected},
            )
            if record is not None:
                candidates.append(record)
    return candidates


def _faithdial_response_is_supported(knowledge: str, response: str) -> bool:
    if _semantic_number_counter(response) - _semantic_number_counter(knowledge):
        return False
    knowledge_tokens = {token.lower() for token in _WORD_RE.findall(knowledge) if len(token) >= 4}
    response_tokens = {token.lower() for token in _WORD_RE.findall(response) if len(token) >= 4}
    if len(knowledge_tokens) < 2 or len(response_tokens) < 2:
        return False
    return len(knowledge_tokens & response_tokens) / len(response_tokens) >= 0.3


def _tar_relative_member(name: str, prefix: str) -> str | None:
    normalized = name.lstrip("./")
    marker = "/" + prefix.strip("/") + "/"
    if marker not in "/" + normalized:
        return None
    return normalized.split(marker, 1)[1]


def _read_tar_jsonl_member(path: Path, requested_member: str) -> Iterable[dict[str, Any]]:
    with tarfile.open(path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.lstrip("./").endswith(requested_member)]
        if len(members) != 1:
            raise PublicCorpusError(f"Expected exactly one tar member ending in {requested_member!r}, found {len(members)}")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise PublicCorpusError(f"Unable to read tar member {requested_member}")
        for row in _read_jsonl(_decoded_lines(handle)):
            yield row


def _decoded_lines(handle: BinaryIO) -> Iterable[str]:
    for line in handle:
        yield line.decode("utf-8")


def _read_jsonl(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise PublicCorpusError(f"Invalid JSONL at line {line_number}: {error}") from error
        if not isinstance(value, dict):
            raise PublicCorpusError(f"JSONL line {line_number} must be an object")
        yield value


def _make_record(
    source: Mapping[str, Any],
    policy_version: str,
    *,
    agent: str,
    user: Any,
    assistant: Any,
    task_type: str,
    transformation: str,
    group_hash: str,
    source_content_sha256: str,
    source_path: str,
    stratum: str,
    language: str,
    quality: Mapping[str, Any],
    tool_ids: Sequence[str] = (),
    preference: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    if agent not in AGENTS:
        raise PublicCorpusError(f"Unsupported target agent: {agent}")
    cleaned_user = _clean_text(user)
    cleaned_assistant = _clean_text(assistant)
    if cleaned_user is None or cleaned_assistant is None:
        return None
    if _contains_pii(cleaned_user) or _contains_pii(cleaned_assistant):
        return None
    cleaned_preference: dict[str, str] | None = None
    if preference is not None:
        chosen = _clean_text(preference.get("chosen"))
        rejected = _clean_text(preference.get("rejected"))
        if (
            chosen is None
            or rejected is None
            or chosen != cleaned_assistant
            or chosen == rejected
            or _contains_pii(chosen)
            or _contains_pii(rejected)
        ):
            return None
        cleaned_preference = {"chosen": chosen, "rejected": rejected}
    if not re.fullmatch(r"[0-9a-f]{64}", source_content_sha256):
        raise PublicCorpusError("source_content_sha256 must be a lowercase SHA-256")
    if not isinstance(source_path, str) or not source_path.strip() or source_path.startswith(("/", "../")):
        raise PublicCorpusError("source_path must be a safe, relative upstream path")
    transformed_content_hash = _sha256_bytes(
        _canonical_json({"assistant": cleaned_assistant, "user": cleaned_user}).encode("utf-8")
    )
    identity_hash = transformed_content_hash
    preference_hash: str | None = None
    if cleaned_preference is not None:
        preference_hash = _sha256_bytes(_canonical_json(cleaned_preference).encode("utf-8"))
        identity_hash = _sha256_bytes(
            f"{transformed_content_hash}\x1f{preference_hash}".encode("utf-8")
        )
    source_family = "public_adapter_corpus_" + re.sub(
        r"[^a-z0-9]+", "_", str(source["id"]).lower()
    ).strip("_")
    record = {
        "schema": RECORD_SCHEMA,
        "id": f"public-{agent}-{identity_hash[:24]}",
        "messages": [
            {"role": "user", "content": cleaned_user},
            {"role": "assistant", "content": cleaned_assistant},
        ],
        "taskType": task_type,
        "sourceFamily": source_family,
        "toolIDs": sorted(set(tool_ids)),
        "metadata": {
            "agent": agent,
            "agentRole": agent,
            "taskType": task_type,
            "sourceFamily": source_family,
            "risk": "standard",
            "publicCorpus": {
                "attribution": source["attribution"],
                "language": language,
                "modified": True,
                "quality": {**quality, "piiRegexScreened": True},
                "sourceArtifactSHA256": source["artifactSHA256"],
                "sourceContentSHA256": source_content_sha256,
                "sourceGroupID": group_hash,
                "sourceID": source["id"],
                "sourceLicense": source["license"],
                "sourceLicenseURL": source["licenseURL"],
                "sourcePath": source_path,
                "sourceRepository": source["datasetID"],
                "sourceRevision": source["revision"],
                "sourceURL": source["sourceURL"],
                "stratum": stratum,
                "targetAdapter": agent,
                "transformation": transformation,
                "transformationVersion": policy_version,
                "transformedContentSHA256": transformed_content_hash,
                "partitionKind": source["partitionKind"],
                "sourcePartition": source["sourcePartition"],
            },
        },
    }
    if cleaned_preference is not None:
        record["preference"] = cleaned_preference
        record["metadata"]["publicCorpus"]["preferenceContentSHA256"] = preference_hash
    return record


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not (MIN_TEXT_CHARS <= len(normalized) <= MAX_TEXT_CHARS):
        return None
    if _CONTROL_RE.search(normalized):
        return None
    return normalized


def _contains_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in (_EMAIL_RE, _PHONE_RE, _SSN_RE, _PAYMENT_CARD_RE, _IPV4_RE, _UUID_RE))


def _opaque_group_hash(*parts: str) -> str:
    return _sha256_bytes("\x1f".join(parts).encode("utf-8"))


def _score_and_select_source_records(
    records: Sequence[dict[str, Any]], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    target_adapters = set(source["targetAdapters"])
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stratum_counts = Counter(
        (
            record["metadata"]["agent"],
            record["metadata"]["publicCorpus"]["stratum"],
        )
        for record in records
    )
    for record in records:
        agent = record["metadata"]["agent"]
        if agent not in target_adapters:
            raise PublicCorpusError(
                f"Transformer {source['transformer']} emitted undeclared target adapter {agent}"
            )
        public = record["metadata"]["publicCorpus"]
        score = _public_record_value_score(
            record,
            source,
            stratum_count=stratum_counts[(agent, public["stratum"])],
        )
        public["selectionScore"] = score
        by_agent[agent].append(record)

    selected: list[dict[str, Any]] = []
    language_caps = source.get("languageCaps")
    for agent in sorted(by_agent):
        candidates = by_agent[agent]
        if isinstance(language_caps, dict):
            positive_caps: dict[str, int] = {}
            for language, language_cap in language_caps.items():
                if not isinstance(language, str) or type(language_cap) is not int or language_cap <= 0:
                    raise PublicCorpusError(
                        f"Source {source['id']} has invalid language cap {language}={language_cap!r}"
                    )
                positive_caps[language] = language_cap
            available_by_language = Counter(
                record["metadata"]["publicCorpus"]["language"] for record in candidates
            )
            eligible_caps = {
                language: language_cap
                for language, language_cap in positive_caps.items()
                if available_by_language[language] > 0
            }
            scale = (
                min(
                    min(1.0, available_by_language[language] / language_cap)
                    for language, language_cap in eligible_caps.items()
                )
                if eligible_caps
                else 1.0
            )
            language_limited: list[dict[str, Any]] = []
            for language, language_cap in sorted(positive_caps.items()):
                language_limited.extend(
                    _value_ranked_group_select(
                        [
                            record
                            for record in candidates
                            if record["metadata"]["publicCorpus"]["language"] == language
                        ],
                        int(language_cap * scale),
                    )
                )
            candidates = language_limited
        selected.extend(_value_ranked_group_select(candidates, source["adapterCaps"][agent]))
    return selected


def _public_record_value_score(
    record: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    stratum_count: int,
) -> dict[str, Any]:
    public = record["metadata"]["publicCorpus"]
    quality = public["quality"]
    profile = source["qualityProfile"]
    source_confidence = _QUALITY_SOURCE_CONFIDENCE[profile]
    transformation_confidence = 0.6
    reasons = [f"quality_profile:{profile}"]
    if quality.get("expertAnnotated") is True:
        transformation_confidence += 0.2
        reasons.append("expert_annotated")
    if quality.get("humanReviewed") is True:
        transformation_confidence += 0.2
        reasons.append("human_reviewed")
    elif quality.get("humanAnnotated") is True:
        transformation_confidence += 0.15
        reasons.append("human_annotated")
    if quality.get("lumenManifestValidated") is True:
        transformation_confidence += 0.15
        reasons.append("manifest_validated")
    if quality.get("semanticAnchorChecked") is True or quality.get("trustedObservationPresent") is True:
        transformation_confidence += 0.1
        reasons.append("semantic_boundary_checked")
    if quality.get("synthetic") is True:
        transformation_confidence -= 0.1
        reasons.append("synthetic_source")
    transformation_confidence = min(1.0, max(0.0, transformation_confidence))

    task_type = str(record.get("taskType") or "")
    boundary_value = 0.7
    if any(token in task_type for token in ("boundary", "repair", "routing", "schema_failure")):
        boundary_value = 1.0
        reasons.append("boundary_case")
    elif any(token in task_type for token in ("tool", "grounded", "preference")):
        boundary_value = 0.9
        reasons.append("role_critical_case")

    messages = record.get("messages") or []
    token_count = sum(len(_WORD_RE.findall(str(message.get("content") or ""))) for message in messages)
    difficulty = min(1.0, 0.45 + token_count / 160.0)
    novelty = min(1.0, 0.65 + 0.35 / max(1, stratum_count))
    if stratum_count <= 2:
        reasons.append("rare_stratum")
    overall = (
        0.25 * source_confidence
        + 0.30 * transformation_confidence
        + 0.20 * boundary_value
        + 0.10 * difficulty
        + 0.15 * novelty
    )
    return {
        "adapterRelevance": round(boundary_value, 6),
        "boundaryValue": round(boundary_value, 6),
        "difficulty": round(difficulty, 6),
        "novelty": round(novelty, 6),
        "overall": round(overall, 6),
        "reasons": sorted(set(reasons)),
        "sourceConfidence": round(source_confidence, 6),
        "transformationConfidence": round(transformation_confidence, 6),
    }


def _value_ranked_group_select(
    records: Sequence[dict[str, Any]], cap: int
) -> list[dict[str, Any]]:
    if cap <= 0:
        return []
    deduplicated: dict[str, dict[str, Any]] = {}
    for record in records:
        content_hash = record["metadata"]["publicCorpus"]["transformedContentSHA256"]
        previous = deduplicated.get(content_hash)
        if previous is None or _record_value_key(record) < _record_value_key(previous):
            deduplicated[content_hash] = record

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in deduplicated.values():
        groups[record["metadata"]["publicCorpus"]["sourceGroupID"]].append(record)
    for items in groups.values():
        items.sort(key=lambda item: item["id"])

    group_order = sorted(
        groups,
        key=lambda group_id: (
            min(_record_value_key(record) for record in groups[group_id]),
            group_id,
        ),
    )
    best_by_stratum: dict[str, str] = {}
    for group_id in group_order:
        for record in groups[group_id]:
            stratum = record["metadata"]["publicCorpus"]["stratum"]
            best_by_stratum.setdefault(stratum, group_id)
    coverage_order = sorted(
        set(best_by_stratum.values()),
        key=lambda group_id: (
            min(_record_value_key(record) for record in groups[group_id]),
            group_id,
        ),
    )

    selected: list[dict[str, Any]] = []
    selected_groups: set[str] = set()
    for group_id in [*coverage_order, *group_order]:
        if group_id in selected_groups:
            continue
        group = groups[group_id]
        if len(selected) + len(group) > cap:
            continue
        selected.extend(group)
        selected_groups.add(group_id)
        if len(selected) == cap:
            break
    return selected


def _record_value_key(record: Mapping[str, Any]) -> tuple[float, str]:
    score = record["metadata"]["publicCorpus"].get("selectionScore") or {}
    overall = score.get("overall") if isinstance(score, dict) else None
    return (-float(overall) if type(overall) in {int, float} else 0.0, str(record.get("id") or ""))


def _stable_balanced_select(records: Sequence[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    if cap <= 0:
        return []
    deduplicated: dict[str, dict[str, Any]] = {}
    for record in records:
        deduplicated.setdefault(record["metadata"]["publicCorpus"]["transformedContentSHA256"], record)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in deduplicated.values():
        strata[record["metadata"]["publicCorpus"]["stratum"]].append(record)
    for items in strata.values():
        items.sort(key=lambda item: item["id"])
    selected: list[dict[str, Any]] = []
    names = sorted(strata)
    index = 0
    while len(selected) < cap:
        made_progress = False
        for name in names:
            items = strata[name]
            if index < len(items):
                selected.append(items[index])
                made_progress = True
                if len(selected) == cap:
                    break
        if not made_progress:
            break
        index += 1
    return selected


def _selection_score_is_valid(value: Any) -> bool:
    numeric_fields = {
        "adapterRelevance",
        "boundaryValue",
        "difficulty",
        "novelty",
        "overall",
        "sourceConfidence",
        "transformationConfidence",
    }
    if not isinstance(value, dict) or set(value) != {*numeric_fields, "reasons"}:
        return False
    if any(type(value.get(field)) not in {int, float} or not 0 <= value[field] <= 1 for field in numeric_fields):
        return False
    reasons = value.get("reasons")
    return (
        isinstance(reasons, list)
        and reasons == sorted(set(reasons))
        and all(isinstance(reason, str) and reason for reason in reasons)
    )


def _validate_records(
    records: Sequence[dict[str, Any]], source_manifest: Mapping[str, Any], contract: LumenContract
) -> None:
    sources = {source["id"]: source for source in source_manifest["sources"]}
    seen_ids: set[str] = set()
    for record in records:
        if record.get("schema") != RECORD_SCHEMA or record.get("id") in seen_ids:
            raise PublicCorpusError(f"Invalid or duplicate public corpus record: {record.get('id')}")
        seen_ids.add(record["id"])
        metadata = record.get("metadata")
        public = metadata.get("publicCorpus") if isinstance(metadata, dict) else None
        if not isinstance(public, dict) or metadata.get("agent") not in AGENTS:
            raise PublicCorpusError(f"Record {record['id']} is missing target-agent provenance")
        source = sources.get(public.get("sourceID"))
        if source is None:
            raise PublicCorpusError(f"Record {record['id']} names an unknown source")
        provenance_pairs = {
            "sourceArtifactSHA256": "artifactSHA256",
            "sourceRepository": "datasetID",
            "sourceLicense": "license",
            "sourceLicenseURL": "licenseURL",
            "sourceRevision": "revision",
            "sourceURL": "sourceURL",
            "partitionKind": "partitionKind",
            "sourcePartition": "sourcePartition",
        }
        for public_key, source_key in provenance_pairs.items():
            if public.get(public_key) != source.get(source_key):
                raise PublicCorpusError(f"Record {record['id']} has mismatched provenance field {public_key}")
        if public.get("sourceLicense") not in source_manifest["allowedLicenses"]:
            raise PublicCorpusError(f"Record {record['id']} has a disallowed license")
        if not isinstance(public.get("sourceURL"), str) or not public["sourceURL"].startswith("https://"):
            raise PublicCorpusError(f"Record {record['id']} requires an HTTPS sourceURL")
        if not re.fullmatch(r"[0-9a-f]{64}", str(public.get("sourceGroupID") or "")):
            raise PublicCorpusError(f"Record {record['id']} requires an opaque SHA-256 sourceGroupID")
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise PublicCorpusError(f"Record {record['id']} requires one user and one assistant message")
        user = messages[0].get("content") if isinstance(messages[0], dict) else None
        assistant = messages[1].get("content") if isinstance(messages[1], dict) else None
        if _clean_text(user) is None or _clean_text(assistant) is None or _contains_pii(user) or _contains_pii(assistant):
            raise PublicCorpusError(f"Record {record['id']} failed text or PII validation")
        expected_content_hash = _sha256_bytes(_canonical_json({"assistant": assistant, "user": user}).encode("utf-8"))
        if not re.fullmatch(r"[0-9a-f]{64}", str(public.get("sourceContentSHA256") or "")):
            raise PublicCorpusError(f"Record {record['id']} source content hash is invalid")
        if public.get("transformedContentSHA256") != expected_content_hash:
            raise PublicCorpusError(f"Record {record['id']} transformed content hash mismatch")
        if public.get("targetAdapter") != metadata["agent"]:
            raise PublicCorpusError(f"Record {record['id']} target adapter does not match metadata.agent")
        if not _selection_score_is_valid(public.get("selectionScore")):
            raise PublicCorpusError(f"Record {record['id']} is missing deterministic selection scoring")
        if metadata["agent"] not in source["targetAdapters"]:
            raise PublicCorpusError(f"Record {record['id']} targets an adapter not declared by its source")
        preference = record.get("preference")
        if preference is not None:
            if (
                not isinstance(preference, dict)
                or _clean_text(preference.get("chosen")) != assistant
                or _clean_text(preference.get("rejected")) is None
                or preference["chosen"] == preference["rejected"]
                or _contains_pii(preference["rejected"])
                or public.get("preferenceContentSHA256")
                != _sha256_bytes(_canonical_json(preference).encode("utf-8"))
            ):
                raise PublicCorpusError(f"Record {record['id']} has an invalid preference pair")
        elif "preferenceContentSHA256" in public:
            raise PublicCorpusError(f"Record {record['id']} has preference provenance without a pair")
        if source["transformer"] == "oasst2_v2":
            serialized = _canonical_json(record)
            if any(field in serialized for field in ("message_id", "parent_id", "user_id", "message_tree_id")):
                raise PublicCorpusError("Raw OpenAssistant identifiers must not be persisted")
        if metadata["agent"] == "executor":
            try:
                call = json.loads(assistant)
            except json.JSONDecodeError as error:
                raise PublicCorpusError(f"Executor record {record['id']} is not JSON") from error
            if not _validate_tool_call(call, contract.tools):
                raise PublicCorpusError(f"Executor record {record['id']} is not a manifest-valid Lumen tool envelope")
            if record.get("toolIDs") != [call["tool"]]:
                raise PublicCorpusError(f"Executor record {record['id']} toolIDs do not match its envelope")


def load_public_adapter_corpus(
    snapshot_dir: Path,
    *,
    lumen_contract: LumenContract,
) -> dict[str, list[dict[str, Any]]]:
    manifest_path = snapshot_dir / "manifest.json"
    records_path = snapshot_dir / "records.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicCorpusError(f"Unable to load snapshot manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise PublicCorpusError("Unsupported public adapter corpus snapshot schema")
    allowed_licenses = manifest.get("allowedLicenses")
    if not isinstance(allowed_licenses, list) or not set(allowed_licenses) <= APPROVED_LICENSES:
        raise PublicCorpusError("Snapshot license policy is invalid")
    if manifest.get("recordsFile") != "records.jsonl":
        raise PublicCorpusError("Snapshot recordsFile must be records.jsonl")
    if manifest.get("lumenContractSHA256") != lumen_contract.sha256:
        raise PublicCorpusError(
            "Snapshot Lumen tool/intent contract does not match the current manifest"
        )
    embedded_source_manifest = {
        "schema": manifest.get("sourceManifestSchema"),
        "selectionPolicyVersion": manifest.get("selectionPolicyVersion"),
        "allowedLicenses": allowed_licenses,
        "sources": manifest.get("sources"),
    }
    _validate_source_manifest(embedded_source_manifest)
    actual_sha = _sha256_file(records_path)
    if actual_sha != manifest.get("recordsSHA256"):
        raise PublicCorpusError(
            f"Snapshot records hash mismatch: expected {manifest.get('recordsSHA256')}, found {actual_sha}"
        )
    records = list(_read_jsonl(records_path.open("rt", encoding="utf-8")))
    if len(records) != manifest.get("recordCount"):
        raise PublicCorpusError("Snapshot record count does not match manifest")
    grouped: dict[str, list[dict[str, Any]]] = {agent: [] for agent in AGENTS}
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    sources = {item["id"]: item for item in manifest.get("sources", []) if isinstance(item, dict) and isinstance(item.get("id"), str)}
    for record in records:
        if record.get("id") in seen_ids:
            raise PublicCorpusError(f"Duplicate snapshot record id: {record.get('id')}")
        seen_ids.add(record.get("id"))
        metadata = record.get("metadata")
        public = metadata.get("publicCorpus") if isinstance(metadata, dict) else None
        agent = metadata.get("agent") if isinstance(metadata, dict) else None
        if agent not in grouped or not isinstance(public, dict):
            raise PublicCorpusError("Snapshot record has invalid target-agent metadata")
        source = sources.get(public.get("sourceID"))
        provenance_pairs = {
            "sourceArtifactSHA256": "artifactSHA256",
            "sourceRepository": "datasetID",
            "sourceLicense": "license",
            "sourceLicenseURL": "licenseURL",
            "sourceRevision": "revision",
            "sourceURL": "sourceURL",
            "partitionKind": "partitionKind",
            "sourcePartition": "sourcePartition",
        }
        if source is None or any(public.get(public_key) != source.get(source_key) for public_key, source_key in provenance_pairs.items()):
            raise PublicCorpusError(f"Snapshot record {record.get('id')} has invalid provenance")
        if public.get("sourceLicense") not in allowed_licenses:
            raise PublicCorpusError(f"Snapshot record {record.get('id')} has a disallowed license")
        if not isinstance(public.get("sourceURL"), str) or not public["sourceURL"].startswith("https://"):
            raise PublicCorpusError(f"Snapshot record {record.get('id')} requires an HTTPS sourceURL")
        if not re.fullmatch(r"[0-9a-f]{64}", str(public.get("sourceGroupID") or "")):
            raise PublicCorpusError(f"Snapshot record {record.get('id')} has an invalid sourceGroupID")
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise PublicCorpusError(f"Snapshot record {record.get('id')} has invalid messages")
        user = messages[0].get("content") if isinstance(messages[0], dict) else None
        assistant = messages[1].get("content") if isinstance(messages[1], dict) else None
        if not isinstance(user, str) or not isinstance(assistant, str):
            raise PublicCorpusError(f"Snapshot record {record.get('id')} has non-string content")
        expected = _sha256_bytes(_canonical_json({"assistant": assistant, "user": user}).encode("utf-8"))
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(public.get("sourceContentSHA256") or ""))
            or public.get("transformedContentSHA256") != expected
            or _contains_pii(user)
            or _contains_pii(assistant)
        ):
            raise PublicCorpusError(f"Snapshot record {record.get('id')} failed content validation")
        if public.get("targetAdapter") != agent:
            raise PublicCorpusError(f"Snapshot record {record.get('id')} target adapter mismatch")
        if not _selection_score_is_valid(public.get("selectionScore")):
            raise PublicCorpusError(f"Snapshot record {record.get('id')} has invalid selection scoring")
        preference = record.get("preference")
        if preference is not None:
            if (
                not isinstance(preference, dict)
                or preference.get("chosen") != assistant
                or not isinstance(preference.get("rejected"), str)
                or preference["chosen"] == preference["rejected"]
                or _contains_pii(preference["rejected"])
                or public.get("preferenceContentSHA256")
                != _sha256_bytes(_canonical_json(preference).encode("utf-8"))
            ):
                raise PublicCorpusError(f"Snapshot record {record.get('id')} has invalid preference data")
        if source["transformer"] == "oasst2_v2" and any(
            field in _canonical_json(record) for field in ("message_id", "parent_id", "user_id", "message_tree_id")
        ):
            raise PublicCorpusError("Raw OpenAssistant identifiers must not be persisted")
        grouped[agent].append(record)
        counts[agent] += 1
    if dict(sorted(counts.items())) != manifest.get("countsByAgent"):
        raise PublicCorpusError("Snapshot agent counts do not match manifest")
    _validate_records(records, embedded_source_manifest, lumen_contract)
    return {agent: items for agent, items in grouped.items() if items}


def _attribution_markdown(source_manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Third-party public adapter datasets",
        "",
        "This snapshot contains filtered, transformed excerpts only; raw source artifacts are not redistributed here.",
        "Every record retains its pinned revision, license, source URL, artifact hash, and content hash.",
        "",
    ]
    for source in source_manifest["sources"]:
        lines.extend(
            [
                f"## {source['datasetID']}",
                "",
                f"- Source: {source['sourceURL']}",
                f"- Revision: `{source['revision']}`",
                f"- Raw artifact SHA-256: `{source['artifactSHA256']}`",
                f"- License: [{source['license']}]({source['licenseURL']})",
                f"- Attribution: {source['attribution']}",
                f"- Lumen transformation: filtered and modified under selection policy `{source_manifest['selectionPolicyVersion']}`.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "AGENTS",
    "PublicCorpusBuildResult",
    "PublicCorpusError",
    "SOURCE_MANIFEST_PATH",
    "acquire_public_corpus_sources",
    "build_public_adapter_corpus",
    "load_lumen_contract",
    "load_public_adapter_corpus",
    "load_public_corpus_source_manifest",
    "lumen_contract_from_manifest",
]
