#!/bin/bash

# Test script for Vespa queries with new schema
# Run this after: just vespa_dev_down vespa_dev_setup vespa_feed_data
# `matches` vs `contains`:
# [1] https://docs.vespa.ai/en/reference/query-language-reference.html

set -e

echo "🔍 Testing Vespa queries with new schema..."
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run query and check result
run_query() {
    local description="$1"
    local query="$2"
    local expected_pattern="$3"
    local extra_params="$4"

    echo -e "${BLUE}Testing: $description${NC}"
    echo "Query: $query $extra_params"

    if [[ -n "$extra_params" ]]; then
        result=$(vespa query "$query" $extra_params 2>/dev/null || echo "QUERY_FAILED")
    else
        result=$(vespa query "$query" 2>/dev/null || echo "QUERY_FAILED")
    fi

    if [[ "$result" == "QUERY_FAILED" ]]; then
        echo -e "${RED}❌ FAILED: Query execution failed${NC}"
        return 1
    fi

    if [[ -n "$expected_pattern" ]] && echo "$result" | grep -q "$expected_pattern"; then
        echo -e "${GREEN}✅ PASSED: Found expected pattern: $expected_pattern${NC}"
    elif [[ -z "$expected_pattern" ]]; then
        echo -e "${GREEN}✅ PASSED: Query executed successfully${NC}"
    else
        echo -e "${RED}❌ FAILED: Expected pattern '$expected_pattern' not found${NC}"
        echo "Result: $result"
        return 1
    fi
    echo
}

# Test basic document retrieval
echo "=== Basic Document Tests ==="
run_query "Get ALL concepts (no filters)" \
    'yql=select * from concept where true' \
    '"totalCount": 5'

run_query "Get all concepts" \
    'yql=select * from concept where true' \
    'q880'

run_query "Get all classifiers" \
    'yql=select * from classifier where true' \
    'kx7m3p9w'

run_query "Get family document with new schema" \
    'yql=select * from family_document where document_import_id contains "CCLW.document.i00000005.n0000"' \
    '"concepts_v2"' \
    'hits=1'

run_query "Get document passage with new schema" \
    'yql=select * from document_passage where family_document_ref contains "id:doc_search:family_document::CCLW.document.i00000005.n0000"' \
    '"concepts_v2"' \
    'hits=1'

# Test concept queries
echo "=== Concept-Specific Tests ==="
run_query "Get air pollution concept" \
    'yql=select * from concept where revisions contains sameElement(value.preferred_label contains "air pollution")' \
    '"q880"'

run_query "Get concept with parents" \
    'yql=select * from concept where revisions contains sameElement(value.subconcept_of_flat matches "q110")' \
    '"q880"'

run_query "Get concept by revision and ID" \
    'yql=select * from concept where revisions contains sameElement(key contains "1402") and id contains "q880"' \
    '"q880"'

# Test *_flat relationship fields
echo "=== Flat Relationship Tests ==="
run_query "Get concepts with subconcept_of_flat q110" \
    'yql=select * from concept where revisions contains sameElement(value.subconcept_of_flat matches "q110")' \
    '"q880"'

run_query "Get concepts with subconcept_of_flat q110:1902" \
    'yql=select * from concept where revisions contains sameElement(value.subconcept_of_flat matches "q110:1902")' \
    '"q880"'

run_query "Get concepts with subconcept_of_flat q110:1902,q290:1903" \
    'yql=select * from concept where revisions contains sameElement(value.subconcept_of_flat matches "q110:1902,q290:1903")' \
    '"q880"'

run_query "Get concepts with subconcept_of_flat q110:1902,q290" \
    'yql=select * from concept where revisions contains sameElement(value.subconcept_of_flat matches "q110:1902,q290")' \
    '"q880"'

run_query "Get concepts with subconcept_of_flat q500" \
    'yql=select * from concept where revisions contains sameElement(value.subconcept_of_flat matches "q500")' \
    '"q730"'

run_query "Get concepts with subconcept_of_flat q500:1904" \
    'yql=select * from concept where revisions contains sameElement(value.subconcept_of_flat contains "q500:1904")' \
    '"q730"'

run_query "Get concepts with subconcept_of_flat q730" \
    'yql=select * from concept where revisions contains sameElement(value.subconcept_of_flat matches "q730")' \
    '"q290"'

run_query "Get concepts with subconcept_of_flat q730:1905" \
    'yql=select * from concept where revisions contains sameElement(value.subconcept_of_flat contains "q730:1905")' \
    '"q290"'

echo -e "${BLUE}Testing: Get multiple concepts by revision and ID pairs${NC}"
echo "Query: yql=select * from concept where (id contains \"q880\" and revisions contains sameElement(key contains \"1402\")) or (id contains \"q290\" and revisions contains sameElement(key contains \"1099\"))"
result=$(vespa query 'yql=select * from concept where (id contains "q880" and revisions contains sameElement(key contains "1402")) or (id contains "q290" and revisions contains sameElement(key contains "1099"))' 2>/dev/null || echo "QUERY_FAILED")

if [[ "$result" == "QUERY_FAILED" ]]; then
    echo -e "${RED}❌ FAILED: Query execution failed${NC}"
elif echo "$result" | grep -q '"q880"' && echo "$result" | grep -q '"q290"'; then
    echo -e "${GREEN}✅ PASSED: Found both q880 and q290 in results${NC}"
else
    echo -e "${RED}❌ FAILED: Expected both q880 and q290 in results${NC}"
    echo "Result: $result"
fi
echo

# Test classifier queries
echo "=== Classifier-Specific Tests ==="
run_query "Get air pollution classifiers" \
    'yql=select * from classifier where name contains "AirPollution"' \
    '"kx7m3p9w"'

# Test new concepts_v2 functionality
echo "=== New Schema Tests ==="
run_query "Family documents with concept q880" \
    'yql=select * from family_document where concepts_v2 contains sameElement(key contains "q880")' \
    '"q880"'

run_query "Family documents with concept q880 and classifier kx7m3p9w" \
    'yql=select * from family_document where concepts_v2 contains sameElement(key contains "q880") and concepts_v2 contains sameElement(value.classifiers_ids_flat matches "kx7m3p9w")' \
    '"q880"'

run_query "Document passages with concept q880" \
    'yql=select * from document_passage where concepts_v2 contains sameElement(key contains "q880")' \
    '"q880"'

run_query "Document passages with concept q880 and classifier kx7m3p9w" \
    'yql=select * from document_passage where concepts_v2 contains sameElement(key contains "q880") and concepts_v2 contains sameElement(value.classifiers_ids_flat matches "kx7m3p9w")' \
    '"q880"'

run_query "Document passages with concept q880 and default classifiers profile" \
    'yql=select * from document_passage where concepts_v2 contains sameElement(key contains "q880")' \
    '"concepts_versions"' \
    'presentation.summary=search_summary'

run_query "Get all classifiers profiles separately (if needed)" \
    'yql=select * from classifiers_profile where true' \
    '"concepts_versions"'

run_query "Get specific classifiers profile by ID (override approach)" \
    'yql=select * from classifiers_profile where id contains "experimentals"' \
    '"experimentals"'

echo "🎉 All tests completed!"
echo

echo "To run individual queries manually:"
echo "# Basic queries:"
echo "vespa query 'yql=select * from concept where true'"
echo "vespa query 'yql=select * from document_passage where concepts_v2 contains sameElement(key contains \"q880\")'"
echo ""
echo "# Single query with default classifiers profile (primaries):"
echo "vespa query 'yql=select * from document_passage where concepts_v2 contains sameElement(key contains \"q880\")' 'presentation.summary=search_summary'"
echo ""
echo "# Separate classifiers profile queries (for overrides):"
echo "vespa query 'yql=select * from classifiers_profile where id contains \"primaries\"'"
echo "vespa query 'yql=select * from classifiers_profile where id contains \"experimentals\"'"
