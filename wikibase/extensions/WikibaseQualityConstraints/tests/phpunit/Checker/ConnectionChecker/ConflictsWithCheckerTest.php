<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\ConnectionChecker;

use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ConflictsWithChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConnectionCheckerHelper;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ConflictsWithChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class ConflictsWithCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var ConnectionCheckerHelper
	 */
	private $connectionCheckerHelper;

	/**
	 * @var ConflictsWithChecker
	 */
	private $checker;

	protected function setUp(): void {
		parent::setUp();
		$this->connectionCheckerHelper = new ConnectionCheckerHelper();
		$this->checker = new ConflictsWithChecker(
			$this->getConstraintParameterParser(),
			$this->connectionCheckerHelper
		);
	}

	public function testConflictsWithConstraintValid() {
		$notConflictingStatement = NewStatement::forProperty( 'P1' )
			->withValue( new ItemId( 'Q42' ) )
			->build();
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( new ItemId( 'Q100' ) )
			->build();
		$entity = NewItem::withId( 'Q4' )
			->andStatement( $notConflictingStatement )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P2' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testConflictsWithConstraintProperty() {
		$notConflictingStatement = NewStatement::forProperty( 'P1' )
			->withValue( new ItemId( 'Q42' ) )
			->build();
		$conflictingStatement = NewStatement::forProperty( 'P2' )
			->withValue( new ItemId( 'Q42' ) )
			->build();
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( new ItemId( 'Q100' ) )
			->build();
		$entity = NewItem::withId( 'Q5' )
			->andStatement( $notConflictingStatement )
			->andStatement( $conflictingStatement )
			->andStatement( $statement )
			->build();

		$constraintParameters = $this->propertyParameter( 'P2' );
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-conflicts-with-property' );
	}

	public function testConflictsWithConstraintPropertyButNotItem() {
		$notConflictingStatement = NewStatement::forProperty( 'P1' )
			->withValue( new ItemId( 'Q42' ) )
			->build();
		$alsoNotConflictingStatement = NewStatement::forProperty( 'P2' )
			->withValue( new ItemId( 'Q42' ) )
			->build();
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( new ItemId( 'Q100' ) )
			->build();
		$entity = NewItem::withId( 'Q5' )
			->andStatement( $notConflictingStatement )
			->andStatement( $alsoNotConflictingStatement )
			->andStatement( $statement )
			->build();

		$constraintParameters = array_merge(
			$this->propertyParameter( 'P2' ),
			$this->itemsParameter( [ 'Q1' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testConflictsWithConstraintPropertyAndItem() {
		$notConflictingStatement = NewStatement::forProperty( 'P1' )
			->withValue( new ItemId( 'Q42' ) )
			->build();
		$conflictingStatement = NewStatement::forProperty( 'P2' )
			->withValue( new ItemId( 'Q42' ) )
			->build();
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( new ItemId( 'Q100' ) )
			->build();
		$entity = NewItem::withId( 'Q5' )
			->andStatement( $notConflictingStatement )
			->andStatement( $conflictingStatement )
			->andStatement( $statement )
			->build();

		$constraintParameters = array_merge(
			$this->propertyParameter( 'P2' ),
			$this->itemsParameter( [ 'Q42' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertViolation( $checkResult, 'wbqc-violation-message-conflicts-with-claim' );
	}

	public function testConflictsWithConstraintPropertyAndNoValue() {
		$notConflictingStatement = NewStatement::forProperty( 'P1' )
			->withValue( new ItemId( 'Q42' ) )
			->build();
		$alsoNotConflictingStatement = NewStatement::noValueFor( 'P2' )
			->build();
		$statement = NewStatement::forProperty( 'P188' )
			->withValue( new ItemId( 'Q100' ) )
			->build();
		$entity = NewItem::withId( 'Q6' )
			->andStatement( $notConflictingStatement )
			->andStatement( $alsoNotConflictingStatement )
			->andStatement( $statement )
			->build();

		$constraintParameters = array_merge(
			$this->propertyParameter( 'P2' ),
			$this->itemsParameter( [ 'Q42' ] )
		);
		$constraint = $this->getConstraintMock( $constraintParameters );

		$checkResult = $this->checker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testConflictsWithConstraintPropertyButDeprecated() {
		$notConflictingStatement = NewStatement::noValueFor( 'P1' )
			->withDeprecatedRank()
			->build();
		$statement = NewStatement::noValueFor( 'P2' )
			->build();
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $notConflictingStatement )
			->andStatement( $statement )
			->build();

		$constraint = $this->getConstraintMock(
			$this->propertyParameter( 'P1' )
		);
		$context = new MainSnakContext( $entity, $statement );

		$checkResult = $this->checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testConflictsWithConstraintPropertyAndItemButDeprecated() {
		$notConflictingStatement = NewStatement::forProperty( 'P1' )
			->withValue( new ItemId( 'Q10' ) )
			->withDeprecatedRank()
			->build();
		$statement = NewStatement::noValueFor( 'P2' )
			->build();
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $notConflictingStatement )
			->andStatement( $statement )
			->build();

		$constraint = $this->getConstraintMock( array_merge(
			$this->propertyParameter( 'P1' ),
			$this->itemsParameter( [ 'Q10' ] )
		) );
		$context = new MainSnakContext( $entity, $statement );

		$checkResult = $this->checker->checkConstraint( $context, $constraint );

		$this->assertCompliance( $checkResult );
	}

	public function testConflictsWithConstraintDeprecatedStatement() {
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
			 ->willReturn( 'Q21502838' );

		return $mock;
	}

}
