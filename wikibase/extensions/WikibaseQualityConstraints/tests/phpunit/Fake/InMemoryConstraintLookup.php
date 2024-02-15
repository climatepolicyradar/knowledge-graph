<?php

namespace WikibaseQuality\ConstraintReport\Tests\Fake;

use Wikibase\DataModel\Entity\NumericPropertyId;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintLookup;
use Wikimedia\Assert\Assert;

/**
 * Simple constraint lookup implentation backed by an array.
 *
 * @license GPL-2.0-or-later
 */
class InMemoryConstraintLookup implements ConstraintLookup {

	/**
	 * @var Constraint[]
	 */
	private $constraints = [];

	/**
	 * @param Constraint[] $constraints
	 */
	public function __construct( array $constraints ) {
		Assert::parameterElementType( Constraint::class, $constraints, '$constraints' );

		$this->constraints = $constraints;
	}

	/**
	 * @param NumericPropertyId $propertyId
	 *
	 * @return Constraint[]
	 */
	public function queryConstraintsForProperty( NumericPropertyId $propertyId ) {
		return array_filter(
			$this->constraints,
			static function ( Constraint $constraint ) use ( $propertyId ) {
				return $constraint->getPropertyId()->equals( $propertyId );
			}
		);
	}

}
