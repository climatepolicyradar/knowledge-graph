<?php

namespace WikibaseQuality\ConstraintReport;

use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikimedia\Rdbms\DBUnexpectedError;

/**
 * @license GPL-2.0-or-later
 */
interface ConstraintStore {

	/**
	 * @param Constraint[] $constraints
	 *
	 * @return bool
	 * @throws DBUnexpectedError
	 */
	public function insertBatch( array $constraints );

	/**
	 * Delete all constraints for the property ID.
	 *
	 * @param NumericPropertyId $propertyId
	 *
	 * @throws DBUnexpectedError
	 */
	public function deleteForProperty( NumericPropertyId $propertyId );

}
