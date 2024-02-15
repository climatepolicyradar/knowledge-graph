<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Cache;

/**
 * A list of entity IDs, along with information whether and how they were cached.
 *
 * (Note that list entries may also be null,
 * in case it was not possible to parse an entity ID from the SPARQL response.)
 *
 * @license GPL-2.0-or-later
 */
class CachedEntityIds extends CachedArray {

	/**
	 * @return array List of EntityId objects. Can contain one or more null values to mark spots
	 *  that should have been EntityIds too, but could not due to errors.
	 */
	public function getArray() {
		return parent::getArray();
	}

}
