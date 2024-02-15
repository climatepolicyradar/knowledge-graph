<?php

namespace WikibaseQuality\ConstraintReport;

use Wikibase\DataModel\Entity\NumericPropertyId;

/**
 * Contains all data belonging to a certain constraint.
 *
 * @license GPL-2.0-or-later
 */
class Constraint {

	/**
	 * @var string
	 */
	private $constraintId;

	/**
	 * @var NumericPropertyId
	 */
	private $propertyId;

	/**
	 * @var string
	 */
	private $constraintTypeItemId;

	/**
	 * @var array
	 */
	private $constraintParameters;

	/**
	 * @param string $constraintId
	 * @param NumericPropertyId $propertyId
	 * @param string $constraintTypeItemId
	 * @param array $constraintParameters
	 */
	public function __construct(
		$constraintId,
		NumericPropertyId $propertyId,
		$constraintTypeItemId,
		array $constraintParameters
	) {
		$this->constraintId = $constraintId;
		$this->propertyId = $propertyId;
		$this->constraintTypeItemId = $constraintTypeItemId;
		$this->constraintParameters = $constraintParameters;
	}

	/**
	 * @return string
	 */
	public function getConstraintId() {
		return $this->constraintId;
	}

	/**
	 * @return string
	 *
	 * Item ID serialization of the constraint type item.
	 */
	public function getConstraintTypeItemId() {
		return $this->constraintTypeItemId;
	}

	/**
	 * @return NumericPropertyId
	 */
	public function getPropertyId() {
		return $this->propertyId;
	}

	/**
	 * The constraint parameters, imported from the qualifiers of the constraint statement.
	 * Contains lists of snak array serializations, indexed by property ID serialization.
	 * (The import is done by {@link UpdateConstraintsTableJob}.)
	 *
	 * @return array
	 */
	public function getConstraintParameters() {
		return $this->constraintParameters;
	}

}
