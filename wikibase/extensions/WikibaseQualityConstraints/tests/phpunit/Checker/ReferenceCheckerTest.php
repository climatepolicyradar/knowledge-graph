<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker;

use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ReferenceChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ReferenceChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ReferenceCheckerTest extends \PHPUnit\Framework\TestCase {

	use ResultAssertions;

	/**
	 * @param string $type context type
	 * @param string|null $messageKey key of violation message, or null if compliance is expected
	 * @dataProvider provideContextTypes
	 */
	public function testReferenceConstraint( $type, $messageKey ) {
		$snak = new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) );
		$context = $this->createMock( Context::class );
		$context->method( 'getType' )->willReturn( $type );
		$context->method( 'getSnak' )->willReturn( $snak );
		$checker = new ReferenceChecker();
		$constraint = $this->getConstraintMock( [] );

		$checkResult = $checker->checkConstraint( $context, $constraint );

		if ( $messageKey === null ) {
			$this->assertCompliance( $checkResult );
		} else {
			$this->assertViolation( $checkResult, $messageKey );
		}
	}

	public static function provideContextTypes() {
		return [
			[ Context::TYPE_STATEMENT, 'wbqc-violation-message-reference' ],
			[ Context::TYPE_QUALIFIER, 'wbqc-violation-message-reference' ],
			[ Context::TYPE_REFERENCE, null ],
		];
	}

	public function testReferenceConstraintDeprecatedStatement() {
		$checker = new ReferenceChecker();
		$statement = NewStatement::noValueFor( 'P1' )
			->withDeprecatedRank()
			->build();
		$constraint = $this->getConstraintMock( [] );
		$entity = NewItem::withId( 'Q1' )
			->build();

		$checkResult = $checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		// this constraint is still checked on deprecated statements
		$this->assertViolation( $checkResult, 'wbqc-violation-message-reference' );
	}

	public function testCheckConstraintParameters() {
		$checker = new ReferenceChecker();
		$constraint = $this->getConstraintMock( [] );

		$result = $checker->checkConstraintParameters( $constraint );

		$this->assertSame( [], $result );
	}

	/**
	 * @return Constraint
	 */
	private function getConstraintMock() {
		$mock = $this->createMock( Constraint::class );
		$mock->method( 'getConstraintParameters' )
			->willReturn( [] );
		$mock->method( 'getConstraintTypeItemId' )
			->willReturn( 'Q21528959' );

		return $mock;
	}

}
