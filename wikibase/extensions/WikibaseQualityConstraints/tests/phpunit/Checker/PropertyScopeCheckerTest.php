<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker;

use Wikibase\DataModel\Reference;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\PropertyScopeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\QualifierContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ReferenceContext;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\PropertyScopeChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class PropertyScopeCheckerTest extends \PHPUnit\Framework\TestCase {
	use ResultAssertions;
	use ConstraintParameters;

	/**
	 * @param Context $context
	 * @param string[] $contextTypes
	 * @param bool $ok
	 * @dataProvider provideContextsAndContextTypes
	 */
	public function testPropertyScopeConstraint( Context $context, array $contextTypes, $ok ) {
		$checker = new PropertyScopeChecker( $this->getConstraintParameterParser() );
		$constraint = $this->getConstraintMock( $this->propertyScopeParameter( $contextTypes ) );

		$result = $checker->checkConstraint( $context, $constraint );

		if ( $ok ) {
			$this->assertCompliance( $result );
		} else {
			$this->assertViolation( $result, 'wbqc-violation-message-property-scope' );
		}
	}

	public static function provideContextsAndContextTypes() {
		$statement = NewStatement::noValueFor( 'P1' )
			->build();
		$snak = $statement->getMainSnak();
		$item = NewItem::withId( 'Q1' )
			->andStatement( $statement )
			->build();
		$statementContext = new MainSnakContext( $item, $statement );
		$qualifierContext = new QualifierContext( $item, $statement, $snak );
		$referenceContext = new ReferenceContext( $item, $statement, new Reference(), $snak );
		$deprecatedStatement = NewStatement::noValueFor( 'P1' )
			->withDeprecatedRank()
			->build();
		$deprecatedContext = new MainSnakContext( $item, $deprecatedStatement );

		return [
			[ $statementContext, [ Context::TYPE_STATEMENT ], true ],
			[ $statementContext, [ Context::TYPE_STATEMENT, Context::TYPE_QUALIFIER ], true ],
			[ $statementContext, [ Context::TYPE_STATEMENT, Context::TYPE_REFERENCE ], true ],
			[ $statementContext, [ Context::TYPE_QUALIFIER, Context::TYPE_REFERENCE ], false ],
			[ $qualifierContext, [ Context::TYPE_QUALIFIER ], true ],
			[ $referenceContext, [ Context::TYPE_REFERENCE ], true ],
			[ $qualifierContext, [ Context::TYPE_REFERENCE ], false ],
			[ $referenceContext, [ Context::TYPE_QUALIFIER ], false ],
			[ $deprecatedContext, [ Context::TYPE_QUALIFIER ], false ],
		];
	}

	public function testCheckConstraintParameters() {
		$checker = new PropertyScopeChecker( $this->getConstraintParameterParser() );
		$constraint = $this->getConstraintMock( [] );

		$result = $checker->checkConstraintParameters( $constraint );

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
			->willReturn( 'Q53869507' );

		return $mock;
	}

}
