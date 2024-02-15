<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\QualifierChecker;

use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Statement\StatementListProvider;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\QualifiersChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\QualifiersChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class QualifiersCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var string[]
	 */
	private $qualifiersList;

	/**
	 * @var QualifiersChecker
	 */
	private $checker;

	protected function setUp(): void {
		parent::setUp();
		$this->qualifiersList = [ 'P580', 'P582', 'P1365', 'P1366', 'P642', 'P805' ];
		$this->checker = new QualifiersChecker( $this->getConstraintParameterParser() );
	}

	/**
	 * @param StatementListProvider $entity
	 *
	 * @return Statement|false
	 */
	private function getFirstStatement( StatementListProvider $entity ) {
		$statements = $entity->getStatements()->toArray();
		return reset( $statements );
	}

	public function testQualifiersConstraint() {
		$entity = NewItem::withId( 'Q2' )
			->andStatement(
				NewStatement::forProperty( 'P39' )
					->withValue( new ItemId( 'Q11696' ) )
					->withQualifier( 'P580', '1970-01-01' )
					->withQualifier( 'P582', 'somevalue' )
					->withQualifier( 'P1365', new ItemId( 'Q207' ) )
			)
			->build();
		$statement = $this->getFirstStatement( $entity );
		$constraint = $this->getConstraintMock( $this->propertiesParameter( $this->qualifiersList ) );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testQualifiersConstraintTooManyQualifiers() {
		$entity = NewItem::withId( 'Q3' )
			->andStatement(
				NewStatement::forProperty( 'P39' )
					->withValue( new ItemId( 'Q11696' ) )
					->withQualifier( 'P580', '1970-01-01' )
					->withQualifier( 'P582', 'somevalue' )
					->withQualifier( 'P1365', new ItemId( 'Q207' ) )
					->withQualifier( 'P39', new ItemId( 'Q11696' ) )
			)
			->build();
		$statement = $this->getFirstStatement( $entity );
		$constraint = $this->getConstraintMock( $this->propertiesParameter( $this->qualifiersList ) );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-qualifiers' );
	}

	public function testQualifiersConstraintNoQualifiers() {
		$entity = NewItem::withId( 'Q4' )
			->andStatement(
				NewStatement::forProperty( 'P39' )
					->withValue( new ItemId( 'Q344' ) )
			)
			->build();
		$statement = $this->getFirstStatement( $entity );
		$constraint = $this->getConstraintMock( $this->propertiesParameter( $this->qualifiersList ) );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testQualifiersConstraintDeprecatedStatement() {
		$statement = NewStatement::noValueFor( 'P1' )
				   ->withDeprecatedRank()
				   ->build();
		$constraint = $this->getConstraintMock( [] );
		$entity = NewItem::withId( 'Q1' )
				->build();

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertDeprecation( $checkResult );
	}

	public function testCheckConstraintParameters() {
		$constraint = $this->getConstraintMock( [] );

		$result = $this->checker->checkConstraintParameters( $constraint );

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
			 ->willReturn( 'Q21510851' );

		return $mock;
	}

}
