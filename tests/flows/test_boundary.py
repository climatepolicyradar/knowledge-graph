import datetime
import json
import re

import pytest
import vespa.querybuilder as qb
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from vespa.exceptions import VespaError
from vespa.io import VespaQueryResponse
from vespa.package import Document, Schema

from flows.boundary import (
    VESPA_MAX_EQUIV_ELEMENTS_IN_QUERY,
    VESPA_MAX_LIMIT,
    DocumentObjectUri,
    TextBlockId,
    convert_labelled_passage_to_concepts,
    get_continuation_tokens_from_query_response,
    get_data_id_from_vespa_hit_id,
    get_document_passage_from_vespa,
    get_document_passages_from_vespa,
    get_document_passages_from_vespa__generator,
    get_parent_concepts_from_concept,
    get_text_block_id_from_vespa_data_id,
    get_vespa_passages_from_query_response,
    get_vespa_search_adapter_from_aws_secrets,
    load_labelled_passages_by_uri,
)
from src.concept import Concept
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span
from tests.flows.conftest import load_fixture

DOCUMENT_PASSAGE_ID_PATTERN = re.compile(
    r"id:doc_search:document_passage::[a-zA-Z]+.[a-zA-Z]+.\d+.\d+.\d+"
)
DATA_ID_PATTERN = re.compile(r"[a-zA-Z]+.[a-zA-Z]+.\d+.\d+.\d+")


def test_get_data_id_from_vespa_hit_id() -> None:
    """Test that we can extract the data ID from a Vespa hit ID."""
    assert (
        DATA_ID_PATTERN.match(
            get_data_id_from_vespa_hit_id(
                "id:doc_search:document_passage::CCLW.executive.00000.0000.001"
            )
        )
        is not None
    )


def test_get_text_block_id_from_vespa_data_id():
    assert (
        get_text_block_id_from_vespa_data_id("CCLW.executive.10014.4470.1273") == "1273"
    )


def test_get_text_block_id_from_vespa_data_id_raises():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "received 6 splits, when expecting 5: ['CCLW', 'executive', '10014', '4470', '1273', '777']"
        ),
    ):
        get_text_block_id_from_vespa_data_id("CCLW.executive.10014.4470.1273.777")


def test_vespa_search_adapter_from_aws_secrets(
    create_vespa_params,
    mock_vespa_credentials,
    tmp_path,
) -> None:
    """Test that we can successfully instantiate the VespaSearchAdpater from ssm params."""
    vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
        cert_dir=str(tmp_path),
        vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
    )

    cert_path = tmp_path / "cert.pem"
    assert cert_path.exists()

    key_path = tmp_path / "key.pem"
    assert key_path.exists()

    with open(cert_path, "r", encoding="utf-8") as f:
        assert f.read() == "Public cert content\n"
    with open(key_path, "r", encoding="utf-8") as f:
        assert f.read() == "Private key content\n"
    assert (
        vespa_search_adapter.instance_url
        == mock_vespa_credentials["VESPA_INSTANCE_URL"]
    )
    assert vespa_search_adapter.client.cert == str(cert_path)
    assert vespa_search_adapter.client.key == str(key_path)


def test_convert_labelled_passges_to_concepts(
    example_labelled_passages: list[LabelledPassage],
) -> None:
    """Test that we can correctly convert labelled passages to concepts."""
    concepts = convert_labelled_passage_to_concepts(example_labelled_passages[0])
    assert all([isinstance(concept, VespaConcept) for concept in concepts])


def test_convert_labelled_passges_to_concepts_skips_invalid_spans(
    example_labelled_passages: list[LabelledPassage],
) -> None:
    """Test that we ignore a Span has no concept ID or timestamps."""
    # Add a span without concept_id
    example_labelled_passages[0].spans.append(
        Span(
            text="Test text.",
            start_index=0,
            end_index=8,
            concept_id=None,
            labellers=[],
        )
    )
    # Add a span without timestamps
    example_labelled_passages[0].spans.append(
        Span(
            text="Test text.",
            start_index=0,
            end_index=8,
            concept_id=WikibaseID("Q123"),
            labellers=[],
            timestamps=[],  # Empty timestamps
        )
    )

    concepts = convert_labelled_passage_to_concepts(example_labelled_passages[0])

    # Verify that the problematic spans were skipped but valid one was processed
    assert len(concepts) == 1


def test_get_parent_concepts_from_concept() -> None:
    """Test that we can correctly retrieve the parent concepts from a concept."""
    assert get_parent_concepts_from_concept(
        concept=Concept(
            preferred_label="forestry sector",
            alternative_labels=[],
            negative_labels=[],
            wikibase_id=WikibaseID("Q10014"),
            subconcept_of=[WikibaseID("Q4470")],
            has_subconcept=[WikibaseID("Q4471")],
            labelled_passages=[],
        )
    ) == ([{"id": "Q4470", "name": ""}], "Q4470,")


@pytest.mark.vespa
def test_get_document_passage_from_vespa(
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
) -> None:
    """Test that we can retrieve a passage for a document in vespa."""

    # Test that we retrieve no passages for a document that doesn't exist
    with pytest.raises(
        ValueError, match="Expected 1 document passage for text block `00001`, got 0"
    ):
        get_document_passage_from_vespa(
            text_block_id="00001",  # This text block doesn't exist
            document_import_id="test.executive.1.1",  # This document doesn't exist
            vespa_search_adapter=local_vespa_search_adapter,
        )

    # Test that we can retrieve all the passages for a document that does exist
    document_passage_id, document_passage = get_document_passage_from_vespa(
        text_block_id="1457",
        document_import_id="CCLW.executive.10014.4470",
        vespa_search_adapter=local_vespa_search_adapter,
    )

    assert isinstance(document_passage, VespaPassage)
    assert isinstance(document_passage_id, str)
    assert DOCUMENT_PASSAGE_ID_PATTERN.fullmatch(document_passage_id)


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_get_some_document_passages_from_vespa(
    local_vespa_search_adapter: VespaSearchAdapter,
    document_passages_test_data_file_path: str,
    vespa_app,
) -> None:
    """Test that we can retrieve some of the passages for a document in vespa."""

    # Test that we retrieve no passages for a document that doesn't exist
    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        document_passages = await get_document_passages_from_vespa(
            text_blocks_ids=["test_33"],
            document_import_id="test.executive.1.1",
            vespa_connection_pool=vespa_connection_pool,
        )

    assert document_passages == []

    # Test that we can retrieve all the passages for a document that does exist
    document_import_id = "CCLW.executive.10014.4470"

    with open(document_passages_test_data_file_path) as f:
        document_passage_test_data = json.load(f)

    text_blocks_ids: list[TextBlockId] = [
        doc["fields"]["text_block_id"]
        for doc in document_passage_test_data
        if doc["fields"]["family_document_ref"]
        == f"id:doc_search:family_document::{document_import_id}"
    ]

    family_document_passages_count_expected = len(text_blocks_ids)

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        document_passages = await get_document_passages_from_vespa(
            document_import_id=document_import_id,
            text_blocks_ids=text_blocks_ids,
            vespa_connection_pool=vespa_connection_pool,
        )

    assert len(document_passages) == family_document_passages_count_expected
    assert all(
        [
            (
                type(passage) is VespaPassage
                and type(passage_id) is str
                and bool(DOCUMENT_PASSAGE_ID_PATTERN.fullmatch(passage_id))
            )
            for passage_id, passage in document_passages
        ]
    )

    # Test that we can retrieve only some of the passages for a document that does exist
    less_expected = 5
    family_document_passages_count_expected = (
        family_document_passages_count_expected - less_expected
    )

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        document_passages = await get_document_passages_from_vespa(
            document_import_id=document_import_id,
            text_blocks_ids=text_blocks_ids[less_expected:],
            vespa_connection_pool=vespa_connection_pool,
        )

    assert len(document_passages) == family_document_passages_count_expected
    assert all(
        [
            (
                type(passage) is VespaPassage
                and type(passage_id) is str
                and bool(DOCUMENT_PASSAGE_ID_PATTERN.fullmatch(passage_id))
            )
            for passage_id, passage in document_passages[less_expected:]
        ]
    )

    # Test that we can construct a query with the configured total text blocks for use
    # in the equivalent operator part of the query.
    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        _ = await get_document_passages_from_vespa(
            document_import_id="test.executive.1.1",
            text_blocks_ids=["test_33"] * VESPA_MAX_EQUIV_ELEMENTS_IN_QUERY,
            vespa_connection_pool=vespa_connection_pool,
        )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_all_get_document_passages_from_vespa(
    local_vespa_search_adapter: VespaSearchAdapter,
    document_passages_test_data_file_path: str,
    vespa_app,
) -> None:
    """Test that we can retrieve all the passages for a document in Vespa."""
    document_import_id = "CCLW.executive.10014.4470"

    with open(document_passages_test_data_file_path) as f:
        document_passage_test_data = json.load(f)

    text_blocks_ids: list[TextBlockId] = [
        doc["fields"]["text_block_id"]
        for doc in document_passage_test_data
        if doc["fields"]["family_document_ref"]
        == f"id:doc_search:family_document::{document_import_id}"
    ]

    family_document_passages_count_expected = len(text_blocks_ids)

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        document_passages = await get_document_passages_from_vespa(
            document_import_id=document_import_id,
            text_blocks_ids=None,
            vespa_connection_pool=vespa_connection_pool,
        )

    assert len(document_passages) == family_document_passages_count_expected
    assert all(
        [
            (
                type(passage) is VespaPassage
                and type(passage_id) is str
                and bool(DOCUMENT_PASSAGE_ID_PATTERN.fullmatch(passage_id))
            )
            for passage_id, passage in document_passages
        ]
    )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_get_document_passages_from_vespa_over_limit(
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
) -> None:
    with pytest.raises(ValueError, match="50001 text block IDs exceeds 50000"):
        async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
            _ = await get_document_passages_from_vespa(
                text_blocks_ids=["test_33"] * (VESPA_MAX_LIMIT + 1),
                document_import_id="test.executive.1.1",
                vespa_connection_pool=vespa_connection_pool,
            )


def test_load_labelled_passages_by_uri_obj(mock_bucket, mock_s3_client):
    fixture = load_fixture("labelled_passages/Q218/v1/AF.document.002MMUCR.n0003.json")
    mock_s3_client.put_object(
        Bucket=mock_bucket,
        Key="labelled_passages/Q218/v1/AF.document.002MMUCR.n0003.json",
        Body=fixture,
        ContentType="application/json",
    )

    document_object_uri: DocumentObjectUri = (
        f"s3://{mock_bucket}/labelled_passages/Q218/v1/AF.document.002MMUCR.n0003.json"
    )
    assert load_labelled_passages_by_uri(document_object_uri) == [
        LabelledPassage(
            id="308",
            text=". Demand Controlled Ventilation via CO2 sensors",
            spans=[
                Span(
                    text=". Demand Controlled Ventilation via CO2 sensors",
                    start_index=36,
                    end_index=39,
                    concept_id="Q218",
                    labellers=['KeywordClassifier("greenhouse gas")'],
                    timestamps=[datetime.datetime(2025, 2, 24, 18, 42, 12, 677997)],
                    id="fxdgkkmc",
                    labelled_text="CO2",
                )
            ],
            metadata={
                "concept": {
                    "preferred_label": "greenhouse gas",
                    "alternative_labels": [
                        "SLCP",
                        "SLCFs",
                        "short lived climate forcers",
                        "hexafluoride",
                        "short-lived climate forcer",
                        "perfluorocarbon",
                        "super pollutants",
                        "nitrogen trifluoride",
                        "CO2",
                        "short-lived climate pollutant",
                        "superpollutants",
                        "HFC",
                        "CFC",
                        "steam",
                        "Hydrofluorocarbon",
                        "CBrClF2",
                        "nitrous",
                        "PFC",
                        "GHG",
                        "freon",
                        "C2H3Cl2",
                        "emission of GHGs",
                        "F-gas",
                        "emissions of GHG",
                        "SLCPs",
                        "HCFCs",
                        "CH3Br",
                        "Halon 1211",
                        "Sulfur hexafluoride",
                        "CFCs",
                        "SFs",
                        "GHG emissions",
                        "carbon tetrachloride",
                        "green house gas emissions",
                        "carbon dioxide equivalent",
                        "aqueous vapor",
                        "carbon gas",
                        "greenhouse gas emissions",
                        "MBr",
                        "nitrous oxide",
                        "hydrochlorofluorocarbons",
                        "HCFC",
                        "CBrF3",
                        "tropospheric ozone",
                        "HFCs",
                        "CO2e",
                        "methyl bromide",
                        "short lived climate pollutants",
                        "GHGs emissions",
                        "CH4",
                        "short lived climate forcer",
                        "green house gases",
                        "SF6",
                        "hfos",
                        "F gas",
                        "methane",
                        "NF3",
                        "short lived climate pollutant",
                        "trifluoride",
                        "ch4",
                        "chlorofluorocarbons",
                        "short-lived climate forcers",
                        "halons",
                        "halon",
                        "carbon dioxide",
                        "CO₂",
                        "PFCs",
                        "Halon 1301",
                        "short-lived climate pollutants",
                        "black carbon",
                        "HFOs",
                        "SF",
                        "carbon tet",
                        "F gases",
                        "N2O",
                        "greenhouse gas",
                        "bromochlorodifluoromethane",
                        "hydrofluoroolefins",
                        "green house gas",
                        "ground-level ozone",
                        "O3",
                        "F-gases",
                        "fluorinated gas",
                        "sulphur hexafluoride",
                        "SLCF",
                        "water vapor",
                        "NFs",
                        "hydrochlorofluorocarbon",
                        "NF",
                        "methyl chloroform",
                        "CH₄",
                        "hydrofluorocarbons",
                        "GHGs",
                        "fluorinated gases",
                        "perfluorocarbons",
                        "bromotrifluoromethane",
                        "chlorofluorocarbon",
                        "greenhouse gases",
                    ],
                    "negative_labels": [],
                    "description": "Greenhouse gases are molecules in our atmosphere that absorb heat radiating from Earth’s surface, preventing it from being emitted into space.",
                    "wikibase_id": "Q218",
                    "subconcept_of": [],
                    "has_subconcept": [
                        "Q223",
                        "Q229",
                        "Q240",
                        "Q237",
                        "Q917",
                        "Q918",
                        "Q919",
                        "Q977",
                        "Q1009",
                        "Q731",
                        "Q732",
                        "Q922",
                        "Q221",
                    ],
                    "related_concepts": [
                        "Q232",
                        "Q560",
                        "Q753",
                        "Q221",
                        "Q978",
                        "Q1652",
                    ],
                    "definition": "Greenhouse gases are molecules in our atmosphere that absorb heat radiating from Earth’s surface, preventing it from being emitted into space. The most common greenhouse gases are (in order of atmospheric concentration) water vapor (H2O), carbon dioxide (CO2), methane (CH4), nitrous oxide (N2O), and a suite of halogen-bearing gases (like fluorocarbons) that are derived from industrial activities.",
                    "labelled_passages": [],
                }
            },
        ),
    ]


def test_load_labelled_passages_by_uri_raw(mock_bucket, mock_s3_client):
    fixture = load_fixture(
        "labelled_passages/Q857/v6/AF.document.i00000021.n0000_translated_en.json"
    )
    mock_s3_client.put_object(
        Bucket=mock_bucket,
        Key="labelled_passages/Q857/v6/AF.document.i00000021.n0000_translated_en.json",
        Body=fixture,
        ContentType="application/json",
    )

    document_object_uri: DocumentObjectUri = f"s3://{mock_bucket}/labelled_passages/Q857/v6/AF.document.i00000021.n0000_translated_en.json"
    assert load_labelled_passages_by_uri(document_object_uri) == [
        LabelledPassage(
            id="58",
            text="13. Projects/programmes supported by the Fund will provide fair, equitable and inclusive access to the expected benefits without hindering access to basic health services, safe water and sanitation, energy, education, housing, safe and decent working conditions and land rights. Projects/programmes should not deepen existing inequalities, particularly those affecting marginalized or vulnerable groups.",
            spans=[
                Span(
                    text="13. Projects/programmes supported by the Fund will provide fair, equitable and inclusive access to the expected benefits without hindering access to basic health services, safe water and sanitation, energy, education, housing, safe and decent working conditions and land rights. Projects/programmes should not deepen existing inequalities, particularly those affecting marginalized or vulnerable groups.",
                    start_index=155,
                    end_index=170,
                    concept_id="Q857",
                    labellers=['KeywordClassifier("healthcare sector")'],
                    timestamps=[datetime.datetime(2025, 2, 25, 1, 35, 56, 388056)],
                    id="b37t5esb",
                    labelled_text="health services",
                )
            ],
            metadata={
                "concept": {
                    "preferred_label": "healthcare sector",
                    "alternative_labels": [
                        "commercial healthcare sector",
                        "specialized hospital services",
                        "pharmaceutical manufacturing sector",
                        "mental health services",
                        "behavioural health service",
                        "family medicine",
                        "complex medical treatment",
                        "health prevention services",
                        "community heal services",
                        "emergency medical services",
                        "manufacture of pharmaceuticals",
                        "EMS",
                        "secondary care sector",
                        "health services industry",
                        "health services",
                        "health rehabilitation service",
                        "medical product distribution",
                        "specialized medical services",
                        "drug service",
                        "pharmaceutical industry",
                        "public medical services",
                        "general healthcare services",
                        "private health service",
                        "public health system",
                        "tertiary care service",
                        "speciality healthcare",
                        "preventive medicine",
                        "public health initiatives",
                        "pharmacy sector",
                        "private medical sector",
                        "public healthcare services management",
                        "specialist healthcare service",
                        "pharmaceutical production",
                        "private healthcare sector",
                        "medical care sector",
                        "alternative medicine",
                        "therapy service",
                        "complementary medicine",
                        "health rehabilitation services",
                        "privatised medical services",
                        "complementary therapy",
                        "private health system",
                        "medical services sector",
                        "pharmaceutical manufacturing",
                        "recovery support service",
                        "health service",
                        "Manufacture of basic pharmaceutical products",
                        "private health services",
                        "rehabilitative care",
                        "specialist healthcare services",
                        "pharmaceutical preparations",
                        "traditional healthcare practices",
                        "medicine production",
                        "urgent medical service",
                        "pharmaceutical care",
                        "healthcare system",
                        "primary healthcare system",
                        "mental wellness programs",
                        "medication management",
                        "public health provision",
                        "mental health service",
                        "integrative medicine",
                        "preventive healthcare",
                        "health sector",
                        "traditional healthcare practice",
                        "medicine industry",
                        "hospital care",
                        "mental healthcare",
                        "primary care services",
                        "medicine manufacturing",
                        "pharmaceutical processing",
                        "urgent medical response",
                        "health rehabilitation program",
                        "public healthcare management",
                        "drug services",
                        "emergency healthcare services",
                        "pharmaceutical sector",
                        "general healthcare service",
                        "clinical specialist services",
                        "complementary therapies",
                        "tertiary healthcare",
                        "traditional medicine",
                        "healthcare services",
                        "public health administration",
                        "secondary healthcare system",
                        "preventive medicine service",
                        "pharmaceutical products manufacturing",
                        "private health industry",
                        "private healthcare system",
                        "advanced medical care",
                        "healthcare industry",
                        "disease prevention strategy",
                    ],
                    "negative_labels": [],
                    "description": "This sector provides medical services, manufactures medical equipment, and develops pharmaceuticals to maintain and improve health.",
                    "wikibase_id": "Q857",
                    "subconcept_of": ["Q709"],
                    "has_subconcept": [
                        "Q1571",
                        "Q1641",
                        "Q1642",
                        "Q1643",
                        "Q1644",
                        "Q1645",
                        "Q1646",
                        "Q1647",
                        "Q806",
                        "Q1648",
                        "Q1650",
                    ],
                    "related_concepts": [
                        "Q1164",
                        "Q418",
                        "Q16",
                        "Q419",
                        "Q974",
                        "Q374",
                        "Q1601",
                        "Q1649",
                    ],
                    "definition": None,
                    "labelled_passages": [],
                }
            },
        )
    ]


def test_get_continuation_tokens_from_query_response(
    mock_vespa_query_response: VespaQueryResponse,
    mock_vespa_query_response_no_continuation_token: VespaQueryResponse,
) -> None:
    """Test that we can get the next continuation tokens from a vespa response."""

    continuation_tokens = get_continuation_tokens_from_query_response(
        mock_vespa_query_response
    )
    assert continuation_tokens == ["BGAAABEDBJGBC"]

    continuation_tokens = get_continuation_tokens_from_query_response(
        mock_vespa_query_response_no_continuation_token
    )
    assert continuation_tokens is None


def test_get_vespa_passages_from_query_response(
    mock_vespa_query_response: VespaQueryResponse,
    mock_vespa_query_response_with_malformed_group: VespaQueryResponse,
) -> None:
    """Test that we can get the passages from a vespa response."""

    passages = get_vespa_passages_from_query_response(mock_vespa_query_response)
    assert len(passages) == 1
    assert isinstance(passages, dict)

    passage = list(passages.values())[0]
    assert isinstance(passage[0], str)
    assert isinstance(passage[1], VespaPassage)

    with pytest.raises(ValueError, match="Vespa passage with no 'fields': "):
        get_vespa_passages_from_query_response(
            mock_vespa_query_response_with_malformed_group
        )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_get_document_passages_from_vespa__generator(
    document_passages_test_data_file_path: str,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
    vespa_lower_max_hit_limit: int,
    vespa_lower_max_hit_limit_query_profile_name: str,
):
    """Test that we can successfully utilise pagination with continuation tokens."""

    grouping_max = 10
    document_import_id = "CCLW.executive.10014.4470"

    with open(document_passages_test_data_file_path) as f:
        document_passage_test_data = json.load(f)

    document_passages_count = len(
        [
            i["fields"]["family_document_ref"]
            for i in document_passage_test_data
            if document_import_id in i["fields"]["family_document_ref"]
        ]
    )

    assert document_passages_count > vespa_lower_max_hit_limit, (
        "the fixture has insufficient document passages to validate the test case"
    )

    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        passages_generator = get_document_passages_from_vespa__generator(
            document_import_id=document_import_id,
            vespa_connection_pool=vespa_connection_pool,
            grouping_max=grouping_max,
            query_profile=vespa_lower_max_hit_limit_query_profile_name,
        )

        responses = []
        async for vespa_passages in passages_generator:
            responses.append(vespa_passages)

        assert len(responses) > 1  # Validate that we did paginate
        for vespa_passages in responses:
            assert isinstance(vespa_passages, dict)
            assert 0 <= len(vespa_passages) <= grouping_max
            for passage in vespa_passages.items():
                assert isinstance(passage[1][0], str)
                assert isinstance(passage[1][1], VespaPassage)

    assert (
        sum(len(vespa_passages) for vespa_passages in responses)
        == document_passages_count
    )


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_lower_max_hits_query_profile(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_lower_max_hit_limit: int,
    vespa_lower_max_hit_limit_query_profile_name: str,
) -> None:
    """Test that we can successfully use the lower_max_hits query profile."""

    hits_within_limit = int(vespa_lower_max_hit_limit / 2)
    hits_beyond_limit = int(vespa_lower_max_hit_limit * 2)

    query: qb.Query = (
        qb.select("*")  # type: ignore
        .from_(
            Schema(name="document_passage", document=Document()),
        )
        .where(True)
    )

    # Confirm that we don't raise below limits
    _: VespaQueryResponse = local_vespa_search_adapter.client.query(
        yql=query,
        queryProfile=vespa_lower_max_hit_limit_query_profile_name,
        hits=hits_within_limit,
    )

    # Confirm that we raise above limits
    with pytest.raises(VespaError) as excinfo:
        local_vespa_search_adapter.client.query(
            yql=query,
            queryProfile=vespa_lower_max_hit_limit_query_profile_name,
            hits=hits_beyond_limit,
        )

    error_info = excinfo.value.args[0][0]
    assert error_info["code"] == 3
    assert error_info["summary"] == "Illegal query"
    assert f"{hits_beyond_limit} hits requested" in error_info["message"]
    assert "configured limit" in error_info["message"]
