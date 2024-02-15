<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use Wikibase\DataModel\Entity\PropertyId;
use Wikibase\DataModel\Snak\Snak;

/**
 * Class for helper functions for value count checkers.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ValueCountCheckerHelper {

	/**
	 * Count the number of snaks with the given property ID.
	 *
	 * @param Snak[] $snaks
	 * @param PropertyId $propertyId
	 * @return int
	 */
	public function getPropertyCount( array $snaks, PropertyId $propertyId ) {
		return count( array_filter(
			$snaks,
			static function ( Snak $snak ) use ( $propertyId ) {
				return $snak->getPropertyId()->equals( $propertyId );
			}
		) );
	}

}
