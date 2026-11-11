-- Does using the term "just transition" more often go with more just transition topics?
-- WHAT YOU GET: one row per UNFCCC submission (section 'unfccc_doc', any year, with its
-- document type) and one row per country across its national laws, policies and UNFCCC
-- submissions (section 'country'), with:
--   passages      paragraphs and list items (>= 20 characters) in documents with classifier output
--   jt            passages using the term (text search for just / fair / equitable transition)
--   nonjt         passages NOT using the term
--   child_*       passages NOT using the term that carry a just transition topic below the root
--                 (validated classifiers Q53, Q58, Q68, Q69, Q1754, Q1744). Counting only
--                 non-term passages removes overlap from labels that contain the phrase itself.
--   ctl_*         comparison topics (validated classifiers), also counted in non-term passages only
-- Unit: passages; UNFCCC submissions by Parties only (author_type 'Party': no CBD/UNCCD, no
-- secretariat, expert-review, IPCC or non-Party stakeholder documents), without emissions inventories. Laws and policies tagged only Nature are left out.
with unfccc as (  -- UNFCCC submissions by Parties (incl. negotiating groups): the principal
                  -- carries author_type 'Party'; documents inherit it through principal_id
  select distinct d.id
  from PRODUCTION.PUBLISHED.DOCUMENTS d
  join PRODUCTION.PUBLISHED.DOCUMENTS pr on pr.id = d.principal_id,
       lateral flatten(input => d.labels) l, lateral flatten(input => pr.labels) a
  where d.category = 'UN submission' and l.value:id::string = 'un_convention::UNFCCC'
    and a.value:id::string = 'author_type::Party'
),
one_country as (  -- documents tagged with exactly one country (geographies also holds regions)
  select d.id, any_value(replace(g.value::string, 'country::', '')) as iso
  from PRODUCTION.PUBLISHED.DOCUMENTS d, lateral flatten(input => d.geographies) g
  where g.value::string like 'country::%'
  group by d.id having count(*) = 1
),
docs as (
  select d.id, d.category, coalesce(d.entity_type, 'Other') as entity_type, year(d.published_date) as yr, oc.iso
  from PRODUCTION.PUBLISHED.DOCUMENTS d
  left join one_country oc on oc.id = d.id
  where not d.is_principal and not d.is_collection and d.document_status = 'published'
    and d.concept_ids is not null
    and d.category in ('UN submission', 'Policy', 'Law')
    and (d.category not in ('Law', 'Policy') or array_contains('Climate'::variant, d.domains))
    and (d.category <> 'UN submission'
         or (d.id in (select id from unfccc)
             and coalesce(d.entity_type, '') <> 'National Inventory Report (NIR)'
             and not (d.title ilike '%inventory%' or d.title ilike '%common reporting table%' or d.title like '%CRT%')))
),
p as (
  select dc.id, dc.category, dc.iso, dc.entity_type, dc.yr, p.concept_ids as c,
         regexp_instr(p.content, '\\b(just|fair|equitable)[[:space:]]+(energy[[:space:]]+|ecological[[:space:]]+)?transitions?\\b|transition juste|transici[óo]n justa|transi[çc][ãa]o justa', 1, 1, 0, 'i') > 0 as jt
  from docs dc
  join PRODUCTION.PUBLISHED.PASSAGES p on p.document_id = dc.id
  where p.content_type in ('Text', 'List') and length(p.content) >= 20
),
f as (
  select id, category, iso, entity_type, yr,
         iff(jt, 1, 0) as jt,
         iff(not jt, 1, 0) as nonjt,
         iff(not jt and (array_contains('Q53'::variant, c) or array_contains('Q58'::variant, c)
             or array_contains('Q68'::variant, c) or array_contains('Q69'::variant, c)
             or array_contains('Q1754'::variant, c) or array_contains('Q1744'::variant, c)), 1, 0) as child_any,
         iff(not jt and array_contains('Q53'::variant, c), 1, 0) as child_socprot,
         iff(not jt and array_contains('Q58'::variant, c), 1, 0) as child_inclusion,
         iff(not jt and array_contains('Q68'::variant, c), 1, 0) as child_decent,
         iff(not jt and array_contains('Q69'::variant, c), 1, 0) as child_greenjobs,
         iff(not jt and array_contains('Q1754'::variant, c), 1, 0) as child_skills,
         iff(not jt and array_contains('Q1744'::variant, c), 1, 0) as child_legal,
         iff(not jt and array_contains('Q557'::variant, c), 1, 0) as ctl_adapt,
         iff(not jt and array_contains('Q385'::variant, c), 1, 0) as ctl_drought,
         iff(not jt and array_contains('Q1286'::variant, c), 1, 0) as ctl_ews,
         iff(not jt and array_contains('Q567'::variant, c), 1, 0) as ctl_renew,
         iff(not jt and array_contains('Q638'::variant, c), 1, 0) as ctl_fossil,
         iff(not jt and array_contains('Q1653'::variant, c), 1, 0) as ctl_netzero,
         iff(not jt and array_contains('Q704'::variant, c), 1, 0) as ctl_women
  from p
)
select 'unfccc_doc' as section, null as unit, null as iso, any_value(yr) as yr,
       any_value(entity_type) as entity_type, count(*) as passages,
       sum(jt) jt, sum(nonjt) nonjt, sum(child_any) child_any, sum(child_socprot) child_socprot,
       sum(child_inclusion) child_inclusion, sum(child_decent) child_decent, sum(child_greenjobs) child_greenjobs,
       sum(child_skills) child_skills, sum(child_legal) child_legal, sum(ctl_adapt) ctl_adapt,
       sum(ctl_drought) ctl_drought, sum(ctl_ews) ctl_ews, sum(ctl_renew) ctl_renew, sum(ctl_fossil) ctl_fossil,
       sum(ctl_netzero) ctl_netzero, sum(ctl_women) ctl_women
from f where category = 'UN submission' group by id
union all
select 'country', iso, iso, null, null, count(*),
       sum(jt), sum(nonjt), sum(child_any), sum(child_socprot), sum(child_inclusion), sum(child_decent),
       sum(child_greenjobs), sum(child_skills), sum(child_legal), sum(ctl_adapt), sum(ctl_drought), sum(ctl_ews),
       sum(ctl_renew), sum(ctl_fossil), sum(ctl_netzero), sum(ctl_women)
from f where iso is not null group by iso
