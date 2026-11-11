-- Chart 5: which economic sectors does just transition content talk about, compared
-- with the corpus as a whole?
-- WHAT YOU GET: one row per (corpus, sector): passages in that corpus and how many carry the
-- sector Topic, and the same two counts for passages tagged with the just transition Topic
-- (Q47, which includes all its sub-topics). Lift (observed / expected) is computed in Python
-- with each corpus's own base rate, so the corpus mix of just transition passages does not
-- drive it.
-- Unit: passages (paragraphs and list items, >= 20 characters) in documents with classifier
-- output, published 2020-2025. UNFCCC submissions by Parties only, without inventories.
-- Laws and policies tagged only Nature are left out.
-- Sectors: the 19 economic-sector classifiers (children of Q709).
with unfccc as (  -- UNFCCC submissions by Parties (incl. negotiating groups): the principal
                  -- carries author_type 'Party'; documents inherit it through principal_id
  select distinct d.id
  from PRODUCTION.PUBLISHED.DOCUMENTS d
  join PRODUCTION.PUBLISHED.DOCUMENTS pr on pr.id = d.principal_id,
       lateral flatten(input => d.labels) l, lateral flatten(input => pr.labels) a
  where d.category = 'UN submission' and l.value:id::string = 'un_convention::UNFCCC'
    and a.value:id::string = 'author_type::Party'
),
p as (
  select case d.category when 'UN submission' then 'UNFCCC submissions'
                         when 'Policy' then 'Policies'
                         when 'Law' then 'Laws'
                         when 'Multilateral Climate Fund project' then 'Climate fund projects'
                         when 'Litigation' then 'Court documents' end as corpus,
         p.concept_ids as c,
         array_contains('Q47'::variant, p.concept_ids) as jt
  from PRODUCTION.PUBLISHED.DOCUMENTS d
  join PRODUCTION.PUBLISHED.PASSAGES p on p.document_id = d.id
  where not d.is_principal and not d.is_collection and d.document_status = 'published'
    and d.concept_ids is not null
    and p.content_type in ('Text', 'List') and length(p.content) >= 20
    and year(d.published_date) between 2020 and 2025
    and d.category in ('UN submission', 'Policy', 'Law', 'Multilateral Climate Fund project', 'Litigation')
    and (d.category not in ('Law', 'Policy') or array_contains('Climate'::variant, d.domains))
    and (d.category <> 'UN submission'
         or (d.id in (select id from unfccc)
             and coalesce(d.entity_type, '') <> 'National Inventory Report (NIR)'
             and not (d.title ilike '%inventory%' or d.title ilike '%common reporting table%' or d.title like '%CRT%')))
),
s as (
  select column1 as qid, column2 as sector from values
    ('Q786', 'Agriculture'), ('Q762', 'Energy supply'), ('Q766', 'Transport and logistics'),
    ('Q779', 'Finance and insurance'), ('Q764', 'Construction'), ('Q775', 'Public sector'),
    ('Q763', 'Environmental management'), ('Q774', 'Education'), ('Q765', 'Trade'),
    ('Q818', 'Water management'), ('Q761', 'Manufacturing'), ('Q787', 'Forestry'),
    ('Q788', 'Fishing'), ('Q760', 'Extractive'), ('Q769', 'ICT'),
    ('Q767', 'Hospitality and tourism'), ('Q768', 'Media'), ('Q856', 'Healthcare'),
    ('Q757', 'Informal economy')
)
select p.corpus, s.qid, s.sector,
       count(*) as passages,
       sum(iff(array_contains(s.qid::variant, p.c), 1, 0)) as passages_sector,
       sum(iff(p.jt, 1, 0)) as jt_passages,
       sum(iff(p.jt and array_contains(s.qid::variant, p.c), 1, 0)) as jt_sector
from p cross join s
group by 1, 2, 3
order by 1, 2
