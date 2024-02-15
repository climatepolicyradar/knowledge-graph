<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Cache;

use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;

/**
 * A list of constraint check results,
 * along with information whether and how they were cached
 * and which other information they depend on.
 *
 * Note that the list of check results may be filtered,
 * and the metadata embedded in this object directly (see {@link getMetadata()})
 * may be more than the merge of the metadata attached to the individual check results.
 * In particular, it is possible for the list of check results to be empty
 * without the metadata being blank.
 *
 * @license GPL-2.0-or-later
 */
class CachedCheckResults extends CachedArray {

	/**
	 * @return CheckResult[] The check results.
	 */
	public function getArray() {
		return parent::getArray();
	}

}
