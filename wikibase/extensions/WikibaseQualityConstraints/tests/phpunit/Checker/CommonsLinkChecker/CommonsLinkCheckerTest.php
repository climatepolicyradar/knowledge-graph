<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\CommonsLinkChecker;

use DataValues\StringValue;
use MediaWiki\Site\MediaWikiPageNameNormalizer;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\Lookup\InMemoryDataTypeLookup;
use Wikibase\DataModel\Services\Lookup\PropertyDataTypeLookup;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\CommonsLinkChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\CommonsLinkChecker
 *
 * @group WikibaseQualityConstraints
 * @group Database
 * @group medium
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class CommonsLinkCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	private const COMMONS_MEDIA_PROPERTY = 'P1';
	private const STRING_PROPERTY = 'P2';
	private const GEO_SHAPE_PROPERTY = 'P3';
	private const TABULAR_DATA_PROPERTY = 'P4';

	/**
	 * @var CommonsLinkChecker
	 */
	private $commonsLinkChecker;

	protected function setUp(): void {
		parent::setUp();
		$pageNameNormalizer = $this->createMock( MediaWikiPageNameNormalizer::class );

		$valueMap = [
			'File:File:test image.jpg' => false,
			'File:test image.jpg' => 'File:Test image.jpg',
			'Category:test category' => 'Category:Test category',
			'test gallery' => 'Test gallery',
			'Creator:Test gallery' => false,
			'Creator:Test creator' => 'Creator:Test creator',
			'Creator:Creator:Test creator' => false,
			'Data:test data' => 'Data:Test data',
			'File:no image.jpg' => false,
		];

		$pageNameNormalizer
			->method( 'normalizePageName' )
			->willReturnCallback( function ( $pageName, $apiUrl ) use ( $valueMap ) {
				$this->assertSame( 'https://commons.wikimedia.org/w/api.php', $apiUrl );
				return $valueMap[$pageName];
			} );

		$this->commonsLinkChecker = new CommonsLinkChecker(
			$this->getConstraintParameterParser(),
			$pageNameNormalizer,
			$this->getFakePropertyDatatypeLookup()
		);
	}

	private function getFakePropertyDatatypeLookup(): PropertyDataTypeLookup {
		$propertyDataTypeLookup = new InMemoryDataTypeLookup();
		$propertyDataTypeLookup->setDataTypeForProperty( new NumericPropertyId( self::COMMONS_MEDIA_PROPERTY ), 'commonsMedia' );
		$propertyDataTypeLookup->setDataTypeForProperty( new NumericPropertyId( self::STRING_PROPERTY ), 'string' );
		$propertyDataTypeLookup->setDataTypeForProperty( new NumericPropertyId( self::GEO_SHAPE_PROPERTY ), 'geo-shape' );
		$propertyDataTypeLookup->setDataTypeForProperty( new NumericPropertyId( self::TABULAR_DATA_PROPERTY ), 'tabular-data' );
		return $propertyDataTypeLookup;
	}

	public function testCommonsLinkConstraintValid() {
		$value = new StringValue( 'test image.jpg' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::COMMONS_MEDIA_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'File' ) )
		);
		$this->assertCompliance( $result );
	}

	public function testCommonsLinkConstraintInvalid() {
		$value1 = new StringValue( 'test_image.jpg' );
		$value2 = new StringValue( 'test%20image.jpg' );
		$value3 = new StringValue( 'File:test image.jpg' );
		$snak1 = new PropertyValueSnak( new NumericPropertyId( self::COMMONS_MEDIA_PROPERTY ), $value1 );
		$snak2 = new PropertyValueSnak( new NumericPropertyId( self::COMMONS_MEDIA_PROPERTY ), $value2 );
		$snak3 = new PropertyValueSnak( new NumericPropertyId( self::COMMONS_MEDIA_PROPERTY ), $value3 );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak1 ),
			$this->getConstraintMock( $this->namespaceParameter( 'File' ) )
		);
		$this->assertViolation( $result, 'wbqc-violation-message-commons-link-not-well-formed' );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak2 ),
			$this->getConstraintMock( $this->namespaceParameter( 'File' ) )
		);
		$this->assertViolation( $result, 'wbqc-violation-message-commons-link-not-well-formed' );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak3 ),
			$this->getConstraintMock( $this->namespaceParameter( 'File' ) )
		);
		$this->assertViolation( $result, 'wbqc-violation-message-commons-link-not-well-formed' );
	}

	public function testCommonsLinkConstraintValidCategory() {
		$value = new StringValue( 'test category' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::COMMONS_MEDIA_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'Category' ) )
		);
		$this->assertCompliance( $result );
	}

	public function testCommonsLinkConstraintValidGallery() {
		$value = new StringValue( 'test gallery' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::COMMONS_MEDIA_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( '' ) )
		);

		$this->assertCompliance( $result );
	}

	public function testCommonsLinkConstraintValidCreator() {
		$value = new StringValue( 'Test creator' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::STRING_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'Creator' ) )
		);

		$this->assertCompliance( $result );
	}

	public function testCommonsLinkConstraintInvalidCreatorExtraNS() {
		$value = new StringValue( 'Creator:Test creator' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::STRING_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'Creator' ) )
		);

		$this->assertViolation( $result, 'wbqc-violation-message-commons-link-not-well-formed' );
	}

	public function testCommonsLinkConstraintInvalidCreatorNonexistent() {
		$value = new StringValue( 'Test gallery' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::STRING_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'Creator' ) )
		);

		$this->assertViolation( $result, 'wbqc-violation-message-commons-link-no-existent' );
	}

	public function testCommonsLinkConstraintValidGeoShape() {
		$value = new StringValue( 'Data:test data' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::GEO_SHAPE_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'Data' ) )
		);

		$this->assertCompliance( $result );
	}

	public function testCommonsLinkConstraintInvalidGeoShapeMissingNS() {
		$value = new StringValue( 'test data' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::GEO_SHAPE_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'Data' ) )
		);

		$this->assertViolation( $result, 'wbqc-violation-message-commons-link-not-well-formed' );
	}

	public function testCommonsLinkConstraintValidTabularData() {
		$value = new StringValue( 'Data:test data' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::TABULAR_DATA_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'Data' ) )
		);

		$this->assertCompliance( $result );
	}

	public function testCommonsLinkConstraintInvalidTabularDataMissingNS() {
		$value = new StringValue( 'test data' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::TABULAR_DATA_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'Data' ) )
		);

		$this->assertViolation( $result, 'wbqc-violation-message-commons-link-not-well-formed' );
	}

	public function testCommonsLinkConstraintNotExistent() {
		$value = new StringValue( 'no image.jpg' );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::COMMONS_MEDIA_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'File' ) )
		);

		$this->assertViolation( $result, 'wbqc-violation-message-commons-link-no-existent' );
	}

	public function testCommonsLinkConstraintNoStringValue() {
		$value = new EntityIdValue( new ItemId( 'Q1' ) );
		$snak = new PropertyValueSnak( new NumericPropertyId( self::COMMONS_MEDIA_PROPERTY ), $value );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'File' ) )
		);

		$this->assertViolation( $result, 'wbqc-violation-message-value-needed-of-type' );
	}

	public function testCommonsLinkConstraintNoValueSnak() {
		$snak = new PropertyNoValueSnak( new NumericPropertyId( self::COMMONS_MEDIA_PROPERTY ) );

		$result = $this->commonsLinkChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->namespaceParameter( 'File' ) )
		);

		$this->assertCompliance( $result );
	}

	public function testCommonsLinkConstraintDeprecatedStatement() {
		$statement = NewStatement::forProperty( self::COMMONS_MEDIA_PROPERTY )
				   ->withValue( 'not_well formed' )
				   ->withDeprecatedRank()
				   ->build();
		$constraint = $this->getConstraintMock( [] );
		$entity = NewItem::withId( 'Q1' )
				->build();

		$checkResult = $this->commonsLinkChecker->checkConstraint( new MainSnakContext( $entity, $statement ), $constraint );

		// this constraint is still checked on deprecated statements
		$this->assertViolation( $checkResult, 'wbqc-violation-message-commons-link-not-well-formed' );
	}

	public function testCheckConstraintParameters() {
		$namespaceId = self::getDefaultConfig()->get( 'WBQualityConstraintsNamespaceId' );
		$constraint = $this->getConstraintMock( [ $namespaceId => [] ] );

		$result = $this->commonsLinkChecker->checkConstraintParameters( $constraint );

		$this->assertCount( 1, $result );
	}

	/**
	 * @param array[] $parameters
	 *
	 * @return Constraint
	 */
	private function getConstraintMock( array $parameters ) {
		$mock = $this->createMock( Constraint::class );
		$mock->method( 'getConstraintParameters' )
			 ->willReturn( $parameters );
		$mock->method( 'getConstraintTypeItemId' )
			 ->willReturn( 'Q21510852' );

		return $mock;
	}

}
