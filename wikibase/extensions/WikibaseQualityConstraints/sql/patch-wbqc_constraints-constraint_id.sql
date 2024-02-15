--
-- T180834: Add numeric primary key to wbqc_constraints
-- (replacing constraint_guid as primary key)

ALTER TABLE /*_*/wbqc_constraints
  ADD COLUMN constraint_id int unsigned NOT NULL AUTO_INCREMENT FIRST,
  DROP PRIMARY KEY,
  ADD PRIMARY KEY (constraint_id);

CREATE INDEX /*i*/wbqc_constraints_guid_index ON /*_*/wbqc_constraints (constraint_guid);
