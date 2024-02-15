<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker;

use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\EntityTypeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\EntityTypeChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author Amir Sarabadani
 * @license GPL-2.0-or-later
 */
class EntityTypeCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var EntityTypeChecker
	 */
	private $entityTypeChecker;

	private $itemEntityType;

	private $propertyEntityType;

	protected function setUp(): void {
		parent::setUp();
		$this->entityTypeChecker = new EntityTypeChecker(
			$this->getConstraintParameterParser()
		);
		$this->itemEntityType = self::getDefaultConfig()->get( 'WBQualityConstraintsWikibaseItemId' );
		$this->propertyEntityType = self::getDefaultConfig()->get( 'WBQualityConstraintsWikibasePropertyId' );
	}

	public function testEntityTypeConstraintValid() {
		$statement = NewStatement::forProperty( 'P123' )
			->withValue( new ItemId( 'Q1' ) )
			->build();

		$result = $this->entityTypeChecker->checkConstraint(
			new FakeSnakContext( $statement->getMainSnak() ),
			$this->getConstraintMock( $this->itemsParameter( [ $this->itemEntityType ] ) )
		);

		$this->assertCompliance( $result );
	}

	public function testEntityTypeConstraintInvalid() {
		$statement = NewStatement::forProperty( 'P123' )
			->withValue( new ItemId( 'Q9' ) )
			->build();

		$result = $this->entityTypeChecker->checkConstraint(
			new FakeSnakContext( $statement->getMainSnak() ),
			$this->getConstraintMock( $this->itemsParameter( [ $this->propertyEntityType ] ) )
		);

		$this->assertViolation( $result, 'wbqc-violation-message-entityType' );
	}

	public function testOneOfConstraintDeprecatedStatement() {
		$statement = NewStatement::noValueFor( 'P1' )
			->withDeprecatedRank()
			->build();
		$constraint = $this->getConstraintMock( [] );
		$entity = NewItem::withId( 'Q1' )
			->build();

		$checkResult = $this->entityTypeChecker->checkConstraint(
			new MainSnakContext( $entity, $statement ),
			$constraint
		);

		$this->assertDeprecation( $checkResult );
	}

	public function testCheckConstraintParameters() {
		$constraint = $this->getConstraintMock( [] );

		$result = $this->entityTypeChecker->checkConstraintParameters( $constraint );

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
			->willReturn( 'Q52004125' );

		return $mock;
	}

}
