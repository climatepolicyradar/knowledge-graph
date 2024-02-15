<?php

namespace WikibaseQuality\ConstraintReport\Tests;

use DataValues\StringValue;
use HashConfig;
use MediaWiki\MediaWikiServices;
use MediaWiki\Site\MediaWikiPageNameNormalizer;
use MockHttpTrait;
use MultiConfig;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\ItemIdParser;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Entity\Property;
use Wikibase\DataModel\Services\Lookup\EntityRetrievingDataTypeLookup;
use Wikibase\DataModel\Services\Lookup\InMemoryEntityLookup;
use Wikibase\DataModel\Services\Lookup\PropertyDataTypeLookup;
use Wikibase\DataModel\Services\Statement\StatementGuidParser;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Snak\SnakList;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use Wikibase\Repo\WikibaseRepo;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\CommonsLinkChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\EntityContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\DelegatingConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\LoggingHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\NullResult;
use WikibaseQuality\ConstraintReport\ConstraintCheckerServices;
use WikibaseQuality\ConstraintReport\ConstraintsServices;
use WikibaseQuality\ConstraintReport\Tests\Fake\InMemoryConstraintLookup;
use WikibaseQuality\ConstraintReport\WikibaseServices;
use Wikimedia\Rdbms\DBUnexpectedError;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\DelegatingConstraintChecker
 *
 * @group WikibaseQualityConstraints
 * @group Database
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class DelegatingConstraintCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;
	use TitleParserMock;
	use MockHttpTrait;

	/**
	 * @var DelegatingConstraintChecker
	 */
	private $constraintChecker;

	/**
	 * @var InMemoryEntityLookup
	 */
	private $lookup;

	/**
	 * Number of constraints for P1.
	 * @var int
	 */
	private $constraintCount;

	protected function setUp(): void {
		parent::setUp();
		$this->installMockHttp( $this->makeFakeHttpRequest( '', 0 ) );
		MediaWikiServices::getInstance()->resetServiceForTesting( ConstraintsServices::CONSTRAINT_LOOKUP );

		$config = new MultiConfig( [
			new HashConfig( [ 'WBQualityConstraintsEnableSuggestionConstraintStatus' => true ] ),
			self::getDefaultConfig(),
			MediaWikiServices::getInstance()->getMainConfig(),
		] );
		$this->setService( 'MainConfig', $config );

		$constraintParameterParser = new ConstraintParameterParser(
			$config,
			WikibaseRepo::getBaseDataModelDeserializerFactory(),
			'http://wikibase.example/entity/'
		);
		$this->setService( ConstraintsServices::CONSTRAINT_PARAMETER_PARSER, $constraintParameterParser );

		$this->lookup = new InMemoryEntityLookup();
		$this->setService( WikibaseServices::ENTITY_LOOKUP, $this->lookup );
		$dataTypeLookup = new EntityRetrievingDataTypeLookup( $this->lookup );
		$this->setService( WikibaseServices::PROPERTY_DATA_TYPE_LOOKUP, $dataTypeLookup );

		$pageNameNormalizer = $this->createMock( MediaWikiPageNameNormalizer::class );
		$pageNameNormalizer->method( 'normalizePageName' )
			->willReturnArgument( 0 );
		$propertyDatatypeLookup = $this->createMock( PropertyDataTypeLookup::class );
		$propertyDatatypeLookup->method( 'getDataTypeIdForProperty' )->willReturn( 'string' );
		$commonsLinkChecker = new CommonsLinkChecker(
			$constraintParameterParser,
			$pageNameNormalizer,
			$propertyDatatypeLookup
		);
		$this->setService( ConstraintCheckerServices::COMMONS_LINK_CHECKER, $commonsLinkChecker );

		$this->constraintChecker = ConstraintsServices::getDelegatingConstraintChecker();

		// specify database tables used by this test
		$this->tablesUsed[] = 'wbqc_constraints';
	}

	protected function tearDown(): void {
		MediaWikiServices::getInstance()->resetServiceForTesting( ConstraintsServices::CONSTRAINT_LOOKUP );
		parent::tearDown();
	}

	/**
	 * @param string $name
	 */
	private function getConstraintTypeItemId( $name ) {
		return self::getDefaultConfig()->get( 'WBQualityConstraints' . $name . 'ConstraintId' );
	}

	/**
	 * Adds temporary test data to database.
	 *
	 * @throws DBUnexpectedError
	 */
	public function addDBData() {
		$config = self::getDefaultConfig();
		$constraints = [
			[
				'constraint_guid' => 'P1$ecb8f617-90f1-4ef3-afab-f4bf3881ec28',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'CommonsLink' ),
				'constraint_parameters' => json_encode(
					$this->namespaceParameter( 'File' )
				),
			],
			[
				'constraint_guid' => 'P10$0bdbe1cb-8afb-4d16-9fd0-c1d0a5b717ce',
				'pid' => 10,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'CommonsLink' ),
				'constraint_parameters' => json_encode( array_merge(
					$this->namespaceParameter( 'File' ),
					$this->exceptionsParameter( [ 'Q5' ] )
				) ),
			],
			[
				'constraint_guid' => 'P11$01c56d1f-b3ce-4a1a-bef7-8c652f395eb2',
				'pid' => 11,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsQualifier' ),
				'constraint_parameters' => json_encode( [
					self::getDefaultConfig()->get( 'WBQualityConstraintsExceptionToConstraintId' ) => [
						[ 'snaktype' => 'novalue', 'property' => 'P2316' ],
					],
				] ),
			],
			[
				'constraint_guid' => 'P1$6ad9eb64-13fd-43a1-afc8-84857108bd59',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'MandatoryQualifier' ),
				'constraint_parameters' => json_encode(
					$this->propertyParameter( 'P2' )
				),
			],
			[
				'constraint_guid' => 'P1$cfff6d73-320c-43c5-8582-e9cbb98e2ca2',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'ConflictsWith' ),
				'constraint_parameters' => json_encode(
					$this->propertyParameter( 'P2' )
				),
			],
			[
				'constraint_guid' => 'P1$c81a981e-4eab-44c9-8aa2-62c63072902e',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'Inverse' ),
				'constraint_parameters' => json_encode(
					$this->propertyParameter( 'P2' )
				),
			],
			[
				'constraint_guid' => 'P1$2040dee1-8c9d-45b7-ac01-2ce8046f578b',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'AllowedQualifiers' ),
				'constraint_parameters' => json_encode(
					$this->propertiesParameter( [ 'P2', 'P3' ] )
				),
			],
			[
				'constraint_guid' => 'P1$09a20b38-fe36-444b-b9ed-22eb46c3ea73',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'DifferenceWithinRange' ),
				'constraint_parameters' => json_encode( array_merge(
					$this->propertyParameter( 'P2' ),
					$this->rangeParameter( 'quantity', 0, 150 )
				) ),
			],
			[
				'constraint_guid' => 'P1$3dac547d-3faf-4198-9b9c-0ba1eae32752',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'Format' ),
				'constraint_parameters' => json_encode(
					$this->formatParameter( '[0-9]' )
				),
			],
			[
				'constraint_guid' => 'P1$cc5708c8-3ec8-4bf3-8931-409530e4d634',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'MultiValue' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$021b2558-8e7c-4c2c-ba14-4596dc11536e',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'DistinctValues' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$3ddc8c54-c056-425c-8745-d257004d585f',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'SingleValue' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$dc4464ed-42a5-47f6-b725-04b1d9d1dfc6',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'Symmetric' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$713ec92d-cd08-413d-b4dc-8e6eeb8c7861',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsQualifier' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$a083d339-7bd6-4737-a987-b55ae8a1a5f3',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'OneOf' ),
				'constraint_parameters' => json_encode(
					$this->itemsParameter( [ 'Q2', 'Q3' ] )
				),
			],
			[
				'constraint_guid' => 'P1$b8587fb1-7315-46ba-9d04-07f0e9af857d',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'Range' ),
				'constraint_parameters' => json_encode(
					$this->rangeParameter( 'time', '0', '2015' )
				),
			],
			[
				'constraint_guid' => 'P1$83ee554c-41fd-4bfa-ae9b-960d0eee2fa4',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'ValueRequiresClaim' ),
				'constraint_parameters' => json_encode( array_merge(
					$this->propertyParameter( 'P2' ),
					$this->itemsParameter( [ 'Q2' ] )
				) ),
			],
			[
				'constraint_guid' => 'P1$370a45b5-b007-455d-b5fa-03b90c629fe5',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'ItemRequiresClaim' ),
				'constraint_parameters' => json_encode( array_merge(
					$this->propertyParameter( 'P2' ),
					$this->itemsParameter( [ 'Q2', 'Q3' ] )
				) ),
			],
			[
				'constraint_guid' => 'P1$831d9d5d-ed77-48f2-8433-fb80a9ef3aad',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'Type' ),
				'constraint_parameters' => json_encode( array_merge(
					$this->relationParameter( 'instance' ),
					$this->classParameter( [ 'Q2', 'Q3' ] )
				) ),
			],
			[
				'constraint_guid' => 'P1$fe667c64-be46-4521-a54d-8a895b6005b0',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'ValueType' ),
				'constraint_parameters' => json_encode( array_merge(
					$this->relationParameter( 'instance' ),
					$this->classParameter( [ 'Q2', 'Q3' ] )
				) ),
			],
			[
				'constraint_guid' => 'P3$0a011ed8-1e2b-470c-a306-fb8ea6953779',
				'pid' => 3,
				'constraint_type_qid' => 'Is not inside',
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P5$2234a1ee-0257-4e13-b619-11211c23734a',
				'pid' => 5,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsQualifier' ),
				'constraint_parameters' => json_encode(
					$this->statusParameter( 'suggestion' )
				),
			],
			[
				'constraint_guid' => 'P6$ad792000-6a12-413d-9fe5-11d2467b7a92',
				'pid' => 6,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsQualifier' ),
				'constraint_parameters' => json_encode(
					$this->statusParameter( 'mandatory' )
				),
			],
			[
				'constraint_guid' => 'P7$a3f746e7-66a0-46fd-96ab-6ff6638332a4',
				'pid' => 7,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsQualifier' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P12$8f54a63f-aea5-4ef5-9415-4343334ca3f1',
				'pid' => 12,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsQualifier' ),
				'constraint_parameters' => json_encode(
					$this->statusParameter( 'invalid' )
				),
			],
			[
				'constraint_guid' => 'P8$34c8af8e-bb50-4458-994b-f355ff899fff',
				'pid' => 8,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsQualifier' ),
				'constraint_parameters' => '{"@error":{"toolong":true}}',
			],
			[
				'constraint_guid' => 'P9$43053ee8-79da-4326-a2ac-f85098291db3',
				'pid' => 9,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsQualifier' ),
				'constraint_parameters' => json_encode( [
					'P2316' => [ [
						'snaktype' => 'novalue',
						'property' => 'P2316',
					] ],
					'P2303' => [ [
						'snaktype' => 'novalue',
						'property' => 'P2303',
					] ],
					'P6607' => [
						$this->constraintClarificationParameter( 'en', 'en clarification 1' )[ 'P6607' ][ 0 ],
						$this->constraintClarificationParameter( 'en', 'en clarification 2' )[ 'P6607' ][ 0 ],
					],
				] ),
			],
			[
				'constraint_guid' => 'P13$c778eee1-2ef9-4826-ab89-389ca6e46b53',
				'pid' => 13,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsQualifier' ),
				'constraint_parameters' => json_encode( array_merge_recursive(
					$this->constraintClarificationParameter( 'de', 'de clarification' ),
					$this->constraintClarificationParameter( 'en', 'en clarification' )
				) ),
			],
			[
				'constraint_guid' => 'P14$9418b916-544f-4f4b-bc4f-c87fb49a5aa2',
				'pid' => 14,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsQualifier' ),
				'constraint_parameters' => json_encode( array_merge_recursive(
					$this->constraintClarificationParameter( 'en', 'en clarification 1' ),
					$this->constraintClarificationParameter( 'en', 'en clarification 2' )
				) ),
			],
			[
				'constraint_guid' => 'P1$a1b1f3d8-6215-4cb6-9edd-3af126ae134f',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedForValuesOnly' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$d7398ac7-aee4-4a29-9e8c-79944e664b67',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'UsedAsReference' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$e5da3fee-1097-49fa-a776-d4e96fe35c0b',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'NoBounds' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$1e2d6650-4249-4ea4-9271-9c95e19b1f41',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'AllowedUnits' ),
				'constraint_parameters' => json_encode(
					$this->itemsParameter( [ 'Q1' ] )
				),
			],
			[
				'constraint_guid' => 'P1$a6970e0d-d67d-4465-b8b6-8debff189332',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'SingleBestValue' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$398c030c-2ed8-4194-9fb1-3e38610f6a31',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'Integer' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$c16e14f7-9f71-4587-9ffd-47790e725cfa',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'CitationNeeded' ),
				'constraint_parameters' => '{}',
			],
			[
				'constraint_guid' => 'P1$6161fca9-71a7-4d5b-9b86-4e717b35ea17',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'PropertyScope' ),
				'constraint_parameters' => json_encode(
					$this->propertyScopeParameter( [ Context::TYPE_STATEMENT ] )
				),
			],
			[
				'constraint_guid' => 'P1$24a3af57-da68-4339-9968-94ded7d99961',
				'pid' => 1,
				'constraint_type_qid' => $this->getConstraintTypeItemId( 'Contemporary' ),
				'constraint_parameters' => '{}',
			],
		];
		$this->constraintCount = count( array_filter(
			$constraints,
			function ( $constraint ) {
				return $constraint['pid'] === 1;
			}
		) );

		$this->db->delete( 'wbqc_constraints', '*' );
		$this->db->insert( 'wbqc_constraints', $constraints );
	}

	public function testCheckOnEntityId() {
		$entity = NewItem::withId( 'Q1' )
			->andStatement(
				NewStatement::forProperty( 'P1' )
					->withValue( 'foo' )
					->withGuid( 'Q1$faa95629-7aa1-48d1-8b4e-a1bcab1b8274' )
			)
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertCount( $this->constraintCount, $result, 'Every constraint should be represented by one result' );
		foreach ( $result as $checkResult ) {
			$constraintTypesWithSparqlErrors = [
				$this->getConstraintTypeItemId( 'DistinctValues' ),
				$this->getConstraintTypeItemId( 'Format' ),
			];
			$constraintType = $checkResult->getConstraint()->getConstraintTypeItemId();
			if ( !in_array( $constraintType, $constraintTypesWithSparqlErrors ) ) {
				$this->assertNotSame( 'todo', $checkResult->getStatus(),
					'Constraints should not be unimplemented' );
			}
			$entityIds = $checkResult->getMetadata()->getDependencyMetadata()->getEntityIds();
			$this->assertContains( $entity->getId(), $entityIds );
			$this->assertContainsEquals( new NumericPropertyId( 'P1' ), $entityIds );
		}
	}

	public function testCheckOnEntityIdEmptyResult() {
		$entity = NewItem::withId( 'Q2' )
			->andStatement(
				NewStatement::forProperty( 'P2' )
					->withValue( 'foo' )
			)
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertSame( [], $result );
	}

	public function testCheckOnEntityIdNullResult() {
		$statement = NewStatement::forProperty( 'P2' )
			->withValue( 'foo' );
		$entity = NewItem::withId( 'Q2' )
			->andStatement( $statement )
			->andStatement( $statement )
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId(
			$entity->getId(),
			null,
			function ( Context $context ) {
				return [ new NullResult( $context->getCursor() ) ];
			}
		);

		$this->assertCount( 2, $result );
	}

	public function testCheckOnEntityIdNoStatements() {
		$entity = NewItem::withId( 'Q2' )->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId(
			$entity->getId(),
			null,
			null,
			function ( EntityId $entityId ) {
				return [ new NullResult( new EntityContextCursor( $entityId->getSerialization() ) ) ];
			}
		);

		$this->assertCount( 1, $result );
	}

	public function testCheckOnEntityIdUnknownConstraint() {
		$entity = NewItem::withId( 'Q3' )
			->andStatement(
				NewStatement::forProperty( 'P3' )
					->withValue( 'foo' )
			)
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertCount( 1, $result );
		$this->assertTodo( $result[0] );
	}

	public function testCheckOnEntityIdNoValue() {
		$entity = NewItem::withId( 'Q4' )
			->andStatement(
				NewStatement::noValueFor( 'P4' )
			)
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertSame( [], $result );
	}

	public function testCheckOnEntityIdKnownException() {
		$entity = NewItem::withId( 'Q5' )
			->andStatement(
				NewStatement::forProperty( 'P10' )
					->withValue( 'foo' )
			)
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertSame( 'exception', $result[ 0 ]->getStatus() );
	}

	public function testCheckOnEntityIdBrokenException() {
		$entity = NewItem::withId( 'Q5' )
			->andStatement( NewStatement::noValueFor( 'P11' ) )
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertSame( 'bad-parameters', $result[ 0 ]->getStatus() );
	}

	public function testCheckOnEntityIdMandatoryConstraint() {
		$entity = NewItem::withId( 'Q6' )
			->andStatement( NewStatement::noValueFor( 'P6' ) )
			->build();
		$this->lookup->addEntity( $entity );

		$results = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertCount( 1, $results );
		$result = $results[0];
		$this->assertViolation( $result );
	}

	public function testCheckOnEntityIdNonMandatoryConstraint() {
		$entity = NewItem::withId( 'Q7' )
			->andStatement( NewStatement::noValueFor( 'P7' ) )
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertSame( 'warning', $result[ 0 ]->getStatus() );
	}

	public function testCheckOnEntityIdSuggestionConstraint() {
		$entity = NewItem::withId( 'Q5' )
			->andStatement( NewStatement::noValueFor( 'P5' ) )
			->build();
		$this->lookup->addEntity( $entity );

		$results = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertCount( 1, $results );
		$result = $results[0];
		$this->assertSame( 'suggestion', $result->getStatus() );
	}

	public function testCheckOnEntityIdInvalidConstraintStatus(): void {
		$entity = NewItem::withId( 'Q12' )
			->andStatement( NewStatement::noValueFor( 'P12' ) )
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertSame( 'bad-parameters', $result[ 0 ]->getStatus() );
		$entityIds = $result[ 0 ]->getMetadata()->getDependencyMetadata()->getEntityIds();
		$this->assertContainsEquals( new NumericPropertyId( 'P12' ), $entityIds );
	}

	public function testCheckOnEntityIdConstraintClarification(): void {
		$entity = NewItem::withId( 'Q13' )
			->andStatement( NewStatement::noValueFor( 'P13' ) )
			->build();
		$this->lookup->addEntity( $entity );

		$results = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertCount( 1, $results );
		$result = $results[0];
		$this->assertSame( [
			[ 'text' => 'de clarification', 'language' => 'de' ],
			[ 'text' => 'en clarification', 'language' => 'en' ],
		], $result->getConstraintClarification()->getArrayValue() );
	}

	public function testCheckOnEntityIdConstraintClarificationInvalid(): void {
		$entity = NewItem::withId( 'Q14' )
			->andStatement( NewStatement::noValueFor( 'P14' ) )
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId( $entity->getId() );

		$this->assertSame( 'bad-parameters', $result[ 0 ]->getStatus() );
		$entityIds = $result[ 0 ]->getMetadata()->getDependencyMetadata()->getEntityIds();
		$this->assertContainsEquals( new NumericPropertyId( 'P14' ), $entityIds );
	}

	public function testCheckOnEntityIdSelectConstraintIds() {
		$entity = NewItem::withId( 'Q1' )
			->andStatement(
				NewStatement::forProperty( 'P1' )
					->withValue( 'foo' )
			)
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnEntityId(
			$entity->getId(),
			[
				'P1$ecb8f617-90f1-4ef3-afab-f4bf3881ec28',
				'P1$6ad9eb64-13fd-43a1-afc8-84857108bd59',
				'P1$cfff6d73-320c-43c5-8582-e9cbb98e2ca2',
			]
		);

		$this->assertCount( 3, $result,
			'Every selected constraint should be represented by one result' );
		foreach ( $result as $checkResult ) {
			$this->assertNotSame( 'todo', $checkResult->getStatus(), 'Constraints should not be unimplemented' );
		}
	}

	public function testCheckOnClaimId() {
		$statement = NewStatement::forProperty( 'P1' )
			->withValue( 'foo' )
			->withGuid( 'Q1$c0f25a6f-9e33-41c8-be34-c86a730ff30b' )
			->build();
		$entity = NewItem::withId( 'Q1' )
			->andStatement( $statement )
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnClaimId(
			$statement->getGuid()
		);

		$this->assertCount( $this->constraintCount, $result,
			'Every constraint should be represented by one result' );
	}

	public function testCheckOnClaimIdEmptyResult() {
		$statement = NewStatement::forProperty( 'P2' )
			->withValue( 'foo' )
			->withGuid( 'Q2$1d1fd258-91ca-4db5-91e4-26219c5aae7a' )
			->build();
		$entity = NewItem::withId( 'Q2' )
			->andStatement( $statement )
			->build();
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkAgainstConstraintsOnClaimId(
			$statement->getGuid()
		);

		$this->assertSame( [], $result );
	}

	public function testCheckOnClaimIdUnknownClaimId() {
		$result = $this->constraintChecker->checkAgainstConstraintsOnClaimId(
			'Q99$does-not-exist' );

		$this->assertSame( [], $result );
	}

	public function testCheckConstraintParametersOnPropertyId() {
		$entity = new Property( new NumericPropertyId( 'P1' ), null, 'time' );
		$this->lookup->addEntity( $entity );

		$result = $this->constraintChecker->checkConstraintParametersOnPropertyId(
			$entity->getId()
		);

		$this->assertCount( $this->constraintCount, $result,
			'Every constraint should be represented by one result' );
		foreach ( $result as $constraintGuid => $constraintResult ) {
			$this->assertSame( [], $constraintResult,
				'Constraint should have no bad parameters' );
		}
	}

	public function testCheckConstraintParametersOnPropertyIdWithError() {
		$result = $this->constraintChecker->checkConstraintParametersOnPropertyId( new NumericPropertyId( 'P8' ) );

		$this->assertCount( 1, $result,
			'Every constraint should be represented by one result' );
		$this->assertCount( 1, $result['P8$34c8af8e-bb50-4458-994b-f355ff899fff'],
			'The constraint should have one exception' );
		$this->assertInstanceOf(
			ConstraintParameterException::class,
			$result['P8$34c8af8e-bb50-4458-994b-f355ff899fff'][0]
		);
	}

	public function testCheckConstraintParametersOnPropertyIdWithMetaErrors() {
		$result = $this->constraintChecker->checkConstraintParametersOnPropertyId( new NumericPropertyId( 'P9' ) );

		$this->assertCount( 1, $result,
			'Every constraint should be represented by one result' );
		$this->assertCount( 3, $result['P9$43053ee8-79da-4326-a2ac-f85098291db3'],
			'The constraint should have three exceptions' );
		$this->assertInstanceOf(
			ConstraintParameterException::class,
			$result['P9$43053ee8-79da-4326-a2ac-f85098291db3'][0]
		);
		$this->assertInstanceOf(
			ConstraintParameterException::class,
			$result['P9$43053ee8-79da-4326-a2ac-f85098291db3'][1]
		);
		$this->assertInstanceOf(
			ConstraintParameterException::class,
			$result['P9$43053ee8-79da-4326-a2ac-f85098291db3'][2]
		);
	}

	public function testCheckConstraintParametersOnConstraintId() {
		$result = $this->constraintChecker->checkConstraintParametersOnConstraintId(
			'P1$ecb8f617-90f1-4ef3-afab-f4bf3881ec28'
		);

		$this->assertSame( [], $result, 'Constraint should exist and have no bad parameters' );
	}

	public function testCheckConstraintParametersOnConstraintIdWhenConstraintDoesNotExist() {
		$result = $this->constraintChecker->checkConstraintParametersOnConstraintId(
			'P1$1735c111-e88c-42f9-8b7a-0692c9c797a3'
		);

		$this->assertNull( $result, 'Constraint should not exist' );
	}

	public function testCheckConstraintParametersOnConstraintIdWithError() {
		$result = $this->constraintChecker->checkConstraintParametersOnConstraintId(
			'P8$34c8af8e-bb50-4458-994b-f355ff899fff'
		);

		$this->assertCount( 1, $result,
			'The constraint should have one exception' );
		$this->assertInstanceOf( ConstraintParameterException::class, $result[0] );
	}

	public function testCheckConstraintParametersOnConstraintIdWithMetaErrors() {
		$result = $this->constraintChecker->checkConstraintParametersOnConstraintId(
			'P9$43053ee8-79da-4326-a2ac-f85098291db3'
		);

		$this->assertCount( 3, $result,
			'The constraint should have three exceptions' );
		$this->assertInstanceOf( ConstraintParameterException::class, $result[0] );
		$this->assertInstanceOf( ConstraintParameterException::class, $result[1] );
		$this->assertInstanceOf( ConstraintParameterException::class, $result[2] );
	}

	public function testPropertiesWithViolatingQualifiers() {
		$q1 = new ItemId( 'Q1' );
		$p2 = new NumericPropertyId( 'P2' );
		$qualifierNotToCheck = new PropertyValueSnak( $p2, new StringValue( 'do not check this ' ) );
		$qualifierToCheck = new PropertyValueSnak( $p2, new StringValue( 'do check this ' ) );
		$entityLookup = new InMemoryEntityLookup();
		$entityLookup->addEntity(
			NewItem::withId( $q1 )
				->andStatement( new Statement(
					new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ),
					new SnakList( [ $qualifierNotToCheck ] )
				) )
				->andStatement( new Statement(
					new PropertyNoValueSnak( new NumericPropertyId( 'P11' ) ),
					new SnakList( [ $qualifierToCheck ] )
				) )
				->build()
		);
		$delegatingConstraintChecker = new DelegatingConstraintChecker(
			$entityLookup,
			[ /* no constraint checkers, status "not implemented" is enough for this test */ ],
			new InMemoryConstraintLookup( [
				new Constraint(
					'P456$a34344b1-1843-4005-bd92-c082d7f7af2f',
					new NumericPropertyId( 'P2' ),
					'Q1',
					[]
				),
			] ),
			$this->getConstraintParameterParser(),
			new StatementGuidParser( new ItemIdParser() ),
			$this->createMock( LoggingHelper::class ),
			true,
			false,
			[ 'P1' ]
		);

		$results = $delegatingConstraintChecker->checkAgainstConstraintsOnEntityId( $q1 );

		$this->assertCount( 1, $results );
		$this->assertSame(
			$qualifierToCheck->getHash(),
			$results[0]->getContextCursor()->getSnakHash()
		);
	}

	public function testSupportedContextTypes_NotSupported() {
		$checker = $this->createMock( ConstraintChecker::class );
		$checker->method( 'getSupportedContextTypes' )
			->willReturn( [
				Context::TYPE_STATEMENT => CheckResult::STATUS_NOT_IN_SCOPE,
			] );
		$checker->method( 'getDefaultContextTypes' )
			->willReturn( [] );
		$checker->method( 'getSupportedEntityTypes' )
			->willReturn( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED );
		$checker->expects( $this->never() )->method( 'checkConstraint' );
		$lookup = new InMemoryEntityLookup();
		$q2 = new ItemId( 'Q2' );
		$lookup->addEntity(
			NewItem::withId( $q2 )
				->andStatement( NewStatement::noValueFor( 'P1' ) )
				->build()
		);
		$delegatingConstraintChecker = new DelegatingConstraintChecker(
			$lookup,
			[ 'Q1' => $checker ],
			new InMemoryConstraintLookup( [
				new Constraint( '', new NumericPropertyId( 'P1' ), 'Q1', [] ),
			] ),
			$this->getConstraintParameterParser(),
			new StatementGuidParser( new ItemIdParser() ),
			$this->createMock( LoggingHelper::class ),
			true,
			true,
			[]
		);

		$results = $delegatingConstraintChecker->checkAgainstConstraintsOnEntityId( $q2 );

		$this->assertCount( 1, $results );
		$this->assertNotInScope( $results[0] );
	}

	public function testSupportedContextTypes_NotImplemented() {
		$checker = $this->createMock( ConstraintChecker::class );
		$checker->method( 'getSupportedContextTypes' )
			->willReturn( [
				Context::TYPE_STATEMENT => CheckResult::STATUS_TODO,
			] );
		$checker->method( 'getDefaultContextTypes' )
			->willReturn( [ Context::TYPE_STATEMENT ] );
		$checker->method( 'getSupportedEntityTypes' )
			->willReturn( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED );
		$checker->expects( $this->never() )->method( 'checkConstraint' );
		$lookup = new InMemoryEntityLookup();
		$q2 = new ItemId( 'Q2' );
		$lookup->addEntity(
			NewItem::withId( $q2 )
				->andStatement( NewStatement::noValueFor( 'P1' ) )
				->build()
		);
		$delegatingConstraintChecker = new DelegatingConstraintChecker(
			$lookup,
			[ 'Q1' => $checker ],
			new InMemoryConstraintLookup( [
				new Constraint( '', new NumericPropertyId( 'P1' ), 'Q1', [] ),
			] ),
			$this->getConstraintParameterParser(),
			new StatementGuidParser( new ItemIdParser() ),
			$this->createMock( LoggingHelper::class ),
			true,
			true,
			[]
		);

		$results = $delegatingConstraintChecker->checkAgainstConstraintsOnEntityId( $q2 );

		$this->assertCount( 1, $results );
		$this->assertTodo( $results[0] );
	}

	public function testSupportedContextTypes_DefaultScope() {
		$checker = $this->createMock( ConstraintChecker::class );
		$checker->method( 'getSupportedContextTypes' )
			->willReturn( [
				Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE,
				Context::TYPE_QUALIFIER => CheckResult::STATUS_COMPLIANCE,
			] );
		$checker->method( 'getDefaultContextTypes' )
			->willReturn( [ Context::TYPE_QUALIFIER ] );
		$checker->method( 'getSupportedEntityTypes' )
			->willReturn( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED );
		$checker->expects( $this->once() )
			->method( 'checkConstraint' )
			->willReturnCallback( function ( Context $context, Constraint $constraint ) {
				$this->assertSame( Context::TYPE_QUALIFIER, $context->getType() );
				return new CheckResult( $context, $constraint );
			} );
		$lookup = new InMemoryEntityLookup();
		$q2 = new ItemId( 'Q2' );
		$lookup->addEntity(
			NewItem::withId( $q2 )
				->andStatement(
					NewStatement::noValueFor( 'P1' )
						->withQualifier( 'P1', 'value' )
				)
				->build()
		);
		$delegatingConstraintChecker = new DelegatingConstraintChecker(
			$lookup,
			[ 'Q1' => $checker ],
			new InMemoryConstraintLookup( [
				new Constraint( '', new NumericPropertyId( 'P1' ), 'Q1', [] ),
			] ),
			$this->getConstraintParameterParser(),
			new StatementGuidParser( new ItemIdParser() ),
			$this->createMock( LoggingHelper::class ),
			true,
			true,
			[]
		);

		$results = $delegatingConstraintChecker->checkAgainstConstraintsOnEntityId( $q2 );
	}

	public function testSupportedContextTypes_ExplicitScope() {
		$checker = $this->createMock( ConstraintChecker::class );
		$checker->method( 'getSupportedContextTypes' )
			->willReturn( [
				Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE,
				Context::TYPE_QUALIFIER => CheckResult::STATUS_COMPLIANCE,
			] );
		$checker->method( 'getDefaultContextTypes' )
			->willReturn( [ Context::TYPE_QUALIFIER ] );
		$checker->method( 'getSupportedEntityTypes' )
			->willReturn( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED );
		$checker->expects( $this->once() )
			->method( 'checkConstraint' )
			->willReturnCallback( function ( Context $context, Constraint $constraint ) {
				$this->assertSame( Context::TYPE_STATEMENT, $context->getType() );
				return new CheckResult( $context, $constraint );
			} );
		$lookup = new InMemoryEntityLookup();
		$q2 = new ItemId( 'Q2' );
		$lookup->addEntity(
			NewItem::withId( $q2 )
				->andStatement(
					NewStatement::noValueFor( 'P1' )
						->withQualifier( 'P1', 'value' )
				)
				->build()
		);
		$delegatingConstraintChecker = new DelegatingConstraintChecker(
			$lookup,
			[ 'Q1' => $checker ],
			new InMemoryConstraintLookup( [
				new Constraint(
					'',
					new NumericPropertyId( 'P1' ),
					'Q1',
					$this->constraintScopeParameter( [ Context::TYPE_STATEMENT ] )
				),
			] ),
			$this->getConstraintParameterParser(),
			new StatementGuidParser( new ItemIdParser() ),
			$this->createMock( LoggingHelper::class ),
			true,
			true,
			[]
		);

		$results = $delegatingConstraintChecker->checkAgainstConstraintsOnEntityId( $q2 );
	}

	public function testSupportedContextTypes_InvalidScope(): void {
		$checker = $this->createMock( ConstraintChecker::class );
		$checker->method( 'getSupportedContextTypes' )
			->willReturn( [
				Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE,
				Context::TYPE_QUALIFIER => CheckResult::STATUS_NOT_IN_SCOPE,
			] );
		$checker->method( 'getDefaultContextTypes' )
			->willReturn( [ Context::TYPE_STATEMENT ] );
		$checker->method( 'getSupportedEntityTypes' )
			->willReturn( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED );
		$checker->expects( $this->never() )
			->method( 'checkConstraint' );
		$lookup = new InMemoryEntityLookup();
		$q2 = new ItemId( 'Q2' );
		$lookup->addEntity(
			NewItem::withId( $q2 )
				->andStatement(
					NewStatement::noValueFor( 'P1' )
						->withQualifier( 'P1', 'value' )
				)
				->build()
		);
		$delegatingConstraintChecker = new DelegatingConstraintChecker(
			$lookup,
			[ 'Q1' => $checker ],
			new InMemoryConstraintLookup( [
				new Constraint(
					'',
					new NumericPropertyId( 'P1' ),
					'Q1',
					$this->constraintScopeParameter( [ Context::TYPE_QUALIFIER ] )
				),
			] ),
			$this->getConstraintParameterParser(),
			new StatementGuidParser( new ItemIdParser() ),
			$this->createMock( LoggingHelper::class ),
			true,
			true,
			[]
		);

		$results = $delegatingConstraintChecker->checkAgainstConstraintsOnEntityId( $q2 );

		$this->assertCount( 2, $results ); // main snak and qualifier
		$this->assertBadParameters( $results[0] );
		$this->assertBadParameters( $results[1] );
	}

	public function testSupportedEntityTypes_NotSupported() {
		$checker = $this->createMock( ConstraintChecker::class );
		$checker->method( 'getSupportedContextTypes' )
			->willReturn( [ Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE ] );
		$checker->method( 'getDefaultContextTypes' )
			->willReturn( [ Context::TYPE_STATEMENT ] );
		$checker->method( 'getSupportedEntityTypes' )
			->willReturn( [ 'item' => CheckResult::STATUS_NOT_IN_SCOPE ] );
		$checker->expects( $this->never() )->method( 'checkConstraint' );
		$lookup = new InMemoryEntityLookup();
		$q2 = new ItemId( 'Q2' );
		$lookup->addEntity(
			NewItem::withId( $q2 )
				->andStatement( NewStatement::noValueFor( 'P1' ) )
				->build()
		);
		$delegatingConstraintChecker = new DelegatingConstraintChecker(
			$lookup,
			[ 'Q1' => $checker ],
			new InMemoryConstraintLookup( [
				new Constraint( '', new NumericPropertyId( 'P1' ), 'Q1', [] ),
			] ),
			$this->getConstraintParameterParser(),
			new StatementGuidParser( new ItemIdParser() ),
			$this->createMock( LoggingHelper::class ),
			true,
			true,
			[]
		);

		$results = $delegatingConstraintChecker->checkAgainstConstraintsOnEntityId( $q2 );

		$this->assertCount( 1, $results );
		$this->assertNotInScope( $results[0] );
	}

	public function testSupportedEntityTypes_NotImplemented() {
		$checker = $this->createMock( ConstraintChecker::class );
		$checker->method( 'getSupportedContextTypes' )
			->willReturn( [ Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE ] );
		$checker->method( 'getDefaultContextTypes' )
			->willReturn( [ Context::TYPE_STATEMENT ] );
		$checker->method( 'getSupportedEntityTypes' )
			->willReturn( [ 'item' => CheckResult::STATUS_TODO ] );
		$checker->expects( $this->never() )->method( 'checkConstraint' );
		$lookup = new InMemoryEntityLookup();
		$q2 = new ItemId( 'Q2' );
		$lookup->addEntity(
			NewItem::withId( $q2 )
				->andStatement( NewStatement::noValueFor( 'P1' ) )
				->build()
		);
		$delegatingConstraintChecker = new DelegatingConstraintChecker(
			$lookup,
			[ 'Q1' => $checker ],
			new InMemoryConstraintLookup( [
				new Constraint( '', new NumericPropertyId( 'P1' ), 'Q1', [] ),
			] ),
			$this->getConstraintParameterParser(),
			new StatementGuidParser( new ItemIdParser() ),
			$this->createMock( LoggingHelper::class ),
			true,
			true,
			[]
		);

		$results = $delegatingConstraintChecker->checkAgainstConstraintsOnEntityId( $q2 );

		$this->assertCount( 1, $results );
		$this->assertTodo( $results[0] );
	}

	public function testSupportedEntityTypes_ExplicitScope() {
		$checker = $this->createMock( ConstraintChecker::class );
		$checker->method( 'getSupportedContextTypes' )
			->willReturn( [
				Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE,
			] );
		$checker->method( 'getDefaultContextTypes' )
			->willReturn( [ Context::TYPE_STATEMENT ] );
		$checker->method( 'getSupportedEntityTypes' )
			->willReturn( ConstraintChecker::ALL_ENTITY_TYPES_SUPPORTED );
		$checker->expects( $this->never() )
			->method( 'checkConstraint' );
		$lookup = new InMemoryEntityLookup();
		$q2 = new ItemId( 'Q2' );
		$lookup->addEntity(
			NewItem::withId( $q2 )
				->andStatement( NewStatement::noValueFor( 'P1' ) )
				->build()
		);
		$delegatingConstraintChecker = new DelegatingConstraintChecker(
			$lookup,
			[ 'Q1' => $checker ],
			new InMemoryConstraintLookup( [
				new Constraint(
					'',
					new NumericPropertyId( 'P1' ),
					'Q1',
					$this->constraintScopeParameter( [], [ 'property' ] )
				),
			] ),
			$this->getConstraintParameterParser(),
			new StatementGuidParser( new ItemIdParser() ),
			$this->createMock( LoggingHelper::class ),
			true,
			true,
			[]
		);

		$results = $delegatingConstraintChecker->checkAgainstConstraintsOnEntityId( $q2 );

		$this->assertCount( 1, $results );
		$this->assertNotInScope( $results[0] );
	}

	public function testSupportedEntityTypes_InvalidScope(): void {
		$checker = $this->createMock( ConstraintChecker::class );
		$checker->method( 'getSupportedContextTypes' )
			->willReturn( [ Context::TYPE_STATEMENT => CheckResult::STATUS_COMPLIANCE ] );
		$checker->method( 'getDefaultContextTypes' )
			->willReturn( [ Context::TYPE_STATEMENT ] );
		$checker->method( 'getSupportedEntityTypes' )
			->willReturn( [ 'item' => CheckResult::STATUS_NOT_IN_SCOPE ] );
		$checker->expects( $this->never() )
			->method( 'checkConstraint' );
		$lookup = new InMemoryEntityLookup();
		$q2 = new ItemId( 'Q2' );
		$lookup->addEntity(
			NewItem::withId( $q2 )
				->andStatement( NewStatement::noValueFor( 'P1' ) )
				->build()
		);
		$delegatingConstraintChecker = new DelegatingConstraintChecker(
			$lookup,
			[ 'Q1' => $checker ],
			new InMemoryConstraintLookup( [
				new Constraint(
					'',
					new NumericPropertyId( 'P1' ),
					'Q1',
					$this->constraintScopeParameter( [], [ 'item' ] )
				),
			] ),
			$this->getConstraintParameterParser(),
			new StatementGuidParser( new ItemIdParser() ),
			$this->createMock( LoggingHelper::class ),
			true,
			true,
			[]
		);

		$results = $delegatingConstraintChecker->checkAgainstConstraintsOnEntityId( $q2 );

		$this->assertCount( 1, $results );
		$this->assertBadParameters( $results[0] );
	}

}
