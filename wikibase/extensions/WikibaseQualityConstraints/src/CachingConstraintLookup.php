<?php

namespace WikibaseQuality\ConstraintReport;

use Wikibase\DataModel\Entity\NumericPropertyId;

/**
 * A ConstraintLookup that caches the results of another lookup in memory
 * (by storing the constraint array per property ID in a big array).
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class CachingConstraintLookup implements ConstraintLookup {

	/**
	 * @var ConstraintLookup
	 */
	private $lookup;

	/**
	 * @var Constraint[][]
	 */
	private $cache = [];

	/**
	 * @param ConstraintLookup $lookup The lookup to which all queries are delegated.
	 */
	public function __construct( ConstraintLookup $lookup ) {
		$this->lookup = $lookup;
	}

	/**
	 * @param NumericPropertyId $propertyId
	 *
	 * @return Constraint[]
	 */
	public function queryConstraintsForProperty( NumericPropertyId $propertyId ) {
		$id = $propertyId->getSerialization();
		if ( !array_key_exists( $id, $this->cache ) ) {
			$this->cache[$id] = $this->lookup->queryConstraintsForProperty( $propertyId );
		}
		return $this->cache[$id];
	}

}
