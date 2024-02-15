<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker;

use DataValues\StringValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertySomeValueSnak;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\NoneOfChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\NoneOfChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author Amir Sarabadani
 * @license GPL-2.0-or-later
 */
class NoneOfCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var NoneOfChecker
	 */
	private $noneOfChecker;

	protected function setUp(): void {
		parent::setUp();
		$this->noneOfChecker = new NoneOfChecker(
			$this->getConstraintParameterParser()
		);
	}

	public function testNoneOfConstraintValid() {
		$statement = NewStatement::forProperty( 'P123' )
			->withValue( new ItemId( 'Q9' ) )
			->build();

		$result = $this->noneOfChecker->checkConstraint(
			new FakeSnakContext( $statement->getMainSnak() ),
			$this->getConstraintMock( $this->itemsParameter( [ 'Q1', 'Q2', 'Q3' ] ) )
		);

		$this->assertCompliance( $result );
	}

	public function testNoneOfConstraintInvalid() {
		$statement = NewStatement::forProperty( 'P123' )
			->withValue( new ItemId( 'Q1' ) )
			->build();

		$result = $this->noneOfChecker->checkConstraint(
			new FakeSnakContext( $statement->getMainSnak() ),
			$this->getConstraintMock( $this->itemsParameter( [ 'Q1', 'Q2', 'Q3' ] ) )
		);

		$this->assertViolation( $result, 'wbqc-violation-message-none-of' );
	}

	public function testNoneOfConstraintWrongType() {
		$statement = NewStatement::forProperty( 'P123' )
			->withValue( new StringValue( 'Q1' ) )
			->build();

		$result = $this->noneOfChecker->checkConstraint(
			new FakeSnakContext( $statement->getMainSnak() ),
			$this->getConstraintMock( $this->itemsParameter( [ 'Q1', 'Q2', 'Q3' ] ) )
		);

		$this->assertCompliance( $result );
	}

	public function testNoneOfConstraintArraySomevalueNovalue() {
		$somevalueSnak = new PropertySomeValueSnak( new NumericPropertyId( 'P123' ) );
		$novalueSnak = new PropertyNoValueSnak( new NumericPropertyId( 'P123' ) );

		foreach ( [ $somevalueSnak, $novalueSnak ] as $allowed ) {
			foreach ( [ $somevalueSnak, $novalueSnak ] as $present ) {
				$result = $this->noneOfChecker->checkConstraint(
					new FakeSnakContext( $present ),
					$this->getConstraintMock( $this->itemsParameter( [ $allowed ] ) )
				);
				if ( $allowed === $present ) {
					$this->assertViolation( $result, 'wbqc-violation-message-none-of' );
				} else {
					$this->assertCompliance( $result );
				}
			}
		}
	}

	public function testNoneOfConstraintDeprecatedStatement() {
		$statement = NewStatement::noValueFor( 'P1' )
			->withDeprecatedRank()
			->build();
		$constraint = $this->getConstraintMock( [] );
		$entity = NewItem::withId( 'Q1' )
			->build();

		$checkResult = $this->noneOfChecker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertDeprecation( $checkResult );
	}

	public function testCheckConstraintParameters() {
		$constraint = $this->getConstraintMock( [] );

		$result = $this->noneOfChecker->checkConstraintParameters( $constraint );

		$this->assertCount( 1, $result );
	}

	/**
	 * @param string[] $parameters
	 *
	 * @return Constraint
	 */
	private function getConstraintMock( array $parameters ) {
		$mock = $this->createMock( Constraint::class );
		$mock->method( 'getConstraintParameters' )
			->willReturn( $parameters );
		$mock->method( 'getConstraintTypeItemId' )
			->willReturn( 'Q52558054' );

		return $mock;
	}

}
