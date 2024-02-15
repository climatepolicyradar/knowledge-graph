--
-- T180834: Add numeric primary key to wbqc_constraints
-- (replacing constraint_guid as primary key)
-- patch-wbqc_constraints-constraint_id.sql failed to make constraint_guid unique,
-- so this patch replaces the index added there with a unique one.
-- The _uniq suffix is necessary for the updater to distinguish between the two versions of the index.

DROP INDEX /*i*/wbqc_constraints_guid_index ON /*_*/wbqc_constraints;
CREATE UNIQUE INDEX /*i*/wbqc_constraints_guid_uniq ON /*_*/wbqc_constraints (constraint_guid);
