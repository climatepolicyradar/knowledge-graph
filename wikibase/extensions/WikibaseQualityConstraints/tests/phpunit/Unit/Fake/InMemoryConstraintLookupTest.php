<?php

namespace WikibaseQuality\ConstraintReport\Tests\Unit\Fake;

use Wikibase\DataModel\Entity\NumericPropertyId;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\Tests\Fake\InMemoryConstraintLookup;

/**
 * @covers WikibaseQuality\ConstraintReport\Tests\Fake\InMemoryConstraintLookup
 *
 * @group WikibaseQualityConstraints
 *
 * @license GPL-2.0-or-later
 */
class InMemoryConstraintLookupTest extends \MediaWikiUnitTestCase {

	public function testQuery_NewLookup_ReturnsEmptyArrayForProperty() {
		$lookup = new InMemoryConstraintLookup( [] );

		$this->assertSame( [], $lookup->queryConstraintsForProperty( new NumericPropertyId( 'P2' ) ) );
	}

	public function testQuery_AddConstraintAndQueryWithItsProperty_ReturnsThatConstraint() {
		$constraint = new Constraint( 'some id', new NumericPropertyId( 'P2' ), 'some type', [] );
		$lookup = new InMemoryConstraintLookup( [ $constraint ] );

		$this->assertSame(
			[ $constraint ],
			$lookup->queryConstraintsForProperty( new NumericPropertyId( 'P2' ) )
		);
	}

	public function testQuery_AddSeveralConstraintsAndQueryForOne_ReturnsOnlyThatConstraint() {
		$expectedConstraint = new Constraint( 'some id', new NumericPropertyId( 'P2' ), 'some type', [] );
		$otherConstraint = new Constraint( 'some id', new NumericPropertyId( 'P3' ), 'some type', [] );
		$lookup = new InMemoryConstraintLookup( [ $expectedConstraint, $otherConstraint ] );

		$this->assertSame(
			[ $expectedConstraint ],
			$lookup->queryConstraintsForProperty( new NumericPropertyId( 'P2' ) )
		);
	}

}
