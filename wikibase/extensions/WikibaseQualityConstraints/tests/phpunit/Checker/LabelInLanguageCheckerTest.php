<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker;

use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Snak\Snak;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\LabelInLanguageChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers \WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\LabelInLanguageChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @license GPL-2.0-or-later
 */
class LabelInLanguageCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	/**
	 * @var LabelInLanguageChecker
	 */
	private $labelInLanguageChecker;

	protected function setUp(): void {
		parent::setUp();
		$this->labelInLanguageChecker = new LabelInLanguageChecker(
			$this->getConstraintParameterParser()
		);
	}

	public function testLabelInLanguageConstraintValid() {
		$statement = NewStatement::forProperty( 'P123' )
			->withValue( new ItemId( 'Q9' ) )
			->build();

		$result = $this->labelInLanguageChecker->checkConstraint(
			$this->getContext( $statement->getMainSnak(), 'en' ),
			$this->getConstraintMock( $this->languageParameter( [ 'en', 'fr' ] ) )
		);

		$this->assertCompliance( $result );
	}

	public function testLabelInLanguageConstraintInvalid() {
		$statement = NewStatement::forProperty( 'P123' )
			->withValue( new ItemId( 'Q1' ) )
			->build();

		$result = $this->labelInLanguageChecker->checkConstraint(
			$this->getContext( $statement->getMainSnak(), 'de' ),
			$this->getConstraintMock( $this->languageParameter( [ 'en', 'fr' ] ) )
		);

		$this->assertViolation( $result, 'wbqc-violation-message-label-lacking' );
	}

	public function testLabelInLanguageConstraintDeprecatedStatement() {
		$statement = NewStatement::noValueFor( 'P1' )
			->withDeprecatedRank()
			->build();
		$constraint = $this->getConstraintMock( [] );
		$item = NewItem::withId( 'Q1' )
			->andLabel( 'en', 'Foo' )
			->build();

		$checkResult = $this->labelInLanguageChecker->checkConstraint(
			new MainSnakContext( $item, $statement ),
			$constraint
		);

		$this->assertDeprecation( $checkResult );
	}

	public function testCheckConstraintParameters() {
		$constraint = $this->getConstraintMock( [] );

		$result = $this->labelInLanguageChecker->checkConstraintParameters( $constraint );

		$this->assertCount( 1, $result );
	}

	/**
	 * @param string[] $parameters
	 *
	 * @return Constraint
	 */
	private function getConstraintMock( array $parameters ) {
		$mock = $this->createMock( Constraint::class );
		$mock->expects( $this->any() )
			->method( 'getConstraintParameters' )
			->willReturn( $parameters );
		$mock->expects( $this->any() )
			->method( 'getConstraintTypeItemId' )
			->willReturn( 'Q108139345' );

		return $mock;
	}

	private function getContext( Snak $snak, string $languageCode ) {
		$item = NewItem::withId( 'Q1' )
			->andLabel( $languageCode, 'Foo' )
			->build();
		return new FakeSnakContext( $snak, $item );
	}

}
