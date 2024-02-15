<?php

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Helper;

use DataValues\UnboundedQuantityValue;
use Wikibase\DataModel\Entity\ItemId;

/**
 * Wrapper class for a constraint parameter representing a list of units.
 *
 * @license GPL-2.0-or-later
 * @author Lucas Werkmeister
 */
class UnitsParameter {

	/**
	 * @var ItemId[]
	 */
	private $unitItemIds;

	/**
	 * @var UnboundedQuantityValue[]
	 */
	private $unitQuantities;

	/**
	 * @var bool
	 */
	private $unitlessAllowed;

	/**
	 * @param ItemId[] $unitItemIds The item IDs of the allowed units.
	 * @param UnboundedQuantityValue[] $unitQuantities Quantities with the allowed units.
	 * @param bool $unitlessAllowed Whether unitless values (unit '1') are allowed or not.
	 */
	public function __construct(
		array $unitItemIds,
		array $unitQuantities,
		$unitlessAllowed
	) {
		$this->unitItemIds = $unitItemIds;
		$this->unitQuantities = $unitQuantities;
		$this->unitlessAllowed = $unitlessAllowed;
	}

	/**
	 * @return ItemId[] The item IDs of the allowed units.
	 */
	public function getUnitItemIds() {
		return $this->unitItemIds;
	}

	/**
	 * @return UnboundedQuantityValue[] Quantities with the allowed units.
	 */
	public function getUnitQuantities() {
		return $this->unitQuantities;
	}

	/**
	 * @return bool Whether unitless values (unit '1') are allowed or not.
	 */
	public function getUnitlessAllowed() {
		return $this->unitlessAllowed;
	}

}
