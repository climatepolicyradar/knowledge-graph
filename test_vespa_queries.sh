#!/bin/bash

# Test script for Vespa queries with new schema
# Run this after: just vespa_feed_data

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
run_query "Get all concepts" \
    'yql=select * from concept where true' \
    'q880'

run_query "Get all models" \
    'yql=select * from model where true' \
    'kx7m3p9w'

run_query "Get family document with new schema" \
    'yql=select * from family_document where document_import_id contains "CCLW.document.i00000005.n0000"' \
    '"concepts_instances"' \
    'hits=1'

run_query "Get document passage with new schema" \
    'yql=select * from document_passage where family_document_ref contains "id:doc_search:family_document::CCLW.document.i00000005.n0000"' \
    '"concepts_instances"' \
    'hits=1'

# Test concept queries
echo "=== Concept-Specific Tests ==="
run_query "Get air pollution concept" \
    'yql=select * from concept where name contains "air pollution"' \
    '"q880"'

run_query "Get concept with parents" \
    'yql=select * from concept where parents contains "q110"' \
    '"q880"'

# Test model queries
echo "=== Model-Specific Tests ==="
run_query "Get air pollution models" \
    'yql=select * from model where name contains "AirPollution"' \
    '"kx7m3p9w"'

# Test new concepts_instances functionality
echo "=== New Schema Tests ==="
run_query "Family documents with concept q880" \
    'yql=select * from family_document where concepts_instances contains sameElement(key contains "q880")' \
    '"q880"'

run_query "Document passages with concept q880" \
    'yql=select * from document_passage where concepts_instances contains sameElement(key contains "q880")' \
    '"q880"'

run_query "Family documents with latest model r5n8qz2t for q880" \
    'yql=select * from family_document where concepts_instances contains sameElement(key contains "q880", value.model_id_latest contains "r5n8qz2t")' \
    '"r5n8qz2t"'

run_query "Document passages with multiple models for q880" \
    'yql=select * from document_passage where concepts_instances contains sameElement(key contains "q880", value.model_id_all matches "kx7m3p9w")' \
    '"kx7m3p9w"'

echo "🎉 All tests completed!"
echo

echo "To run individual queries manually:"
echo "vespa query 'yql=select * from concept where true'"
echo "vespa query 'yql=select * from family_document where concepts_instances contains sameElement(key contains \"q880\")'"
echo "vespa query 'yql=select * from document_passage where concepts_instances contains sameElement(key contains \"q880\", value.model_id_latest contains \"r5n8qz2t\")'"
