-- Charts 2 and 2b: one row per country (UNFCCC Party as tagged in `geographies`, the EU as one Party).
-- WHAT YOU GET: for each country, whether it has an NDC in each round, whether any of
-- that round's NDC files names a just transition (text search for the term, *_jt), and
-- whether any carries the just transition topic (Q47 classifier output, *_topic).
-- NDC rounds are approximated by publication year: 2015-19 (first NDCs), 2020-24
-- (updated / second NDCs), 2025-26 (NDC 3.0). All files of a round are pooled.
with p as (
  select replace(g.value::string, 'country::', '') as iso,
         year(d.published_date) as yr,
         regexp_instr(p.content, '\\b(just|fair|equitable)[[:space:]]+(energy[[:space:]]+|ecological[[:space:]]+)?transitions?\\b|transition juste|transici[óo]n justa|transi[çc][ãa]o justa', 1, 1, 0, 'i') > 0 as jt,
         array_contains('Q47'::variant, p.concept_ids) as topic
  from PRODUCTION.PUBLISHED.DOCUMENTS d
  join PRODUCTION.PUBLISHED.PASSAGES p on p.document_id = d.id,
       lateral flatten(input => d.geographies) g
  where not d.is_principal and not d.is_collection and d.document_status = 'published'
    and length(p.content) >= 20
    and g.value::string like 'country::%'
    and d.entity_type = 'Nationally Determined Contribution (NDC)'
),
ndc as (
  select iso,
    max(iff(yr between 2015 and 2019, 1, 0)) as ndc1, max(iff(yr between 2015 and 2019 and jt, 1, 0)) as ndc1_jt,
    max(iff(yr between 2020 and 2024, 1, 0)) as ndc2, max(iff(yr between 2020 and 2024 and jt, 1, 0)) as ndc2_jt,
    max(iff(yr >= 2025, 1, 0)) as ndc3,               max(iff(yr >= 2025 and jt, 1, 0)) as ndc3_jt,
    max(iff(yr between 2015 and 2019 and topic, 1, 0)) as ndc1_topic,
    max(iff(yr between 2020 and 2024 and topic, 1, 0)) as ndc2_topic,
    max(iff(yr >= 2025 and topic, 1, 0)) as ndc3_topic
  from p group by 1
)
select 'iso,ndc1,ndc1_jt,ndc2,ndc2_jt,ndc3,ndc3_jt,ndc1_topic,ndc2_topic,ndc3_topic' || '|' ||
       listagg(iso || ',' || ndc1 || ',' || ndc1_jt || ',' || ndc2 || ',' || ndc2_jt || ',' || ndc3 || ',' || ndc3_jt
               || ',' || ndc1_topic || ',' || ndc2_topic || ',' || ndc3_topic, '|')
         within group (order by iso) as csv
from ndc
