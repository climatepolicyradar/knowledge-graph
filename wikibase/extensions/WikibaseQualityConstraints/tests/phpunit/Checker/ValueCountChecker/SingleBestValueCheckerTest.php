<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\ValueCountChecker;

use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\SingleBestValueChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\SingleBestValueChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class SingleBestValueCheckerTest extends \PHPUnit\Framework\TestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var SingleBestValueChecker
	 */
	private $checker;

	protected function setUp(): void {
		parent::setUp();

		$this->checker = new SingleBestValueChecker( $this->getConstraintParameterParser() );
	}

	/**
	 * @dataProvider provideRanksAndMessageKeys
	 */
	public function testCheckConstraint( array $ranks, $messageKey ) {
		$item = NewItem::withId( 'Q1' );
		foreach ( $ranks as $rank ) {
			$statement = NewStatement::noValueFor( 'P1' )
				->withRank( $rank )
				->build();
			$item = $item->andStatement( $statement );
		}
		$context = new MainSnakContext( $item->build(), $statement );
		$constraint = $this->getConstraintMock();

		$checkResult = $this->checker->checkConstraint( $context, $constraint );

		if ( $messageKey !== null ) {
			$this->assertViolation( $checkResult, $messageKey );
		} else {
			$this->assertCompliance( $checkResult );
		}
	}

	public static function provideRanksAndMessageKeys() {
		$normal = Statement::RANK_NORMAL;
		$preferred = Statement::RANK_PREFERRED;

		return [
			[ [ $normal ], null ],
			[ [ $preferred ], null ],
			[ [ $normal, $preferred ], null ],
			[ [ $normal, $normal, $preferred ], null ],
			[ [ $normal, $normal ], 'wbqc-violation-message-single-best-value-no-preferred' ],
			[ [ $normal, $preferred, $preferred ], 'wbqc-violation-message-single-best-value-multi-preferred' ],
		];
	}

	public function testCheckConstraint_Separators() {
		$statement1 = NewStatement::noValueFor( 'P1' )
			->withQualifier( 'P2', 'foo' )
			->build();
		$statement2 = NewStatement::noValueFor( 'P1' )
			->withQualifier( 'P2', 'bar' )
			->build();
		$item = NewItem::withId( 'Q1' )
			->andStatement( $statement1 )
			->andStatement( $statement2 )
			->build();
		$context = new MainSnakContext( $item, $statement1 );
		$constraint = $this->getConstraintMock(
			$this->separatorsParameter( [ 'P2' ] )
		);

		$checkResult = $this->checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testCheckConstraint_Deprecated() {
		$statement = NewStatement::noValueFor( 'P1' )
			->withDeprecatedRank()
			->build();
		$entity = NewItem::withId( 'Q1' )
			->build();
		$context = new MainSnakContext( $entity, $statement );
		$constraint = $this->getConstraintMock();

		$checkResult = $this->checker->checkConstraint( $context, $constraint );

		$this->assertDeprecation( $checkResult );
	}

	public function testCheckConstraintParameters() {
		$constraint = $this->getConstraintMock();

		$result = $this->checker->checkConstraintParameters( $constraint );

		$this->assertSame( [], $result );
	}

	/**
	 * @param array $parameters
	 *
	 * @return Constraint
	 */
	private function getConstraintMock( array $parameters = [] ) {
		$mock = $this->createMock( Constraint::class );
		$mock->method( 'getConstraintParameters' )
			->willReturn( $parameters );
		$mock->method( 'getConstraintTypeItemId' )
			->willReturn( 'Q52060874' );

		return $mock;
	}

}
