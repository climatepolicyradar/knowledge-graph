<?php

namespace WikibaseQuality\ConstraintReport\Tests\Specials;

use DataValues\StringValue;
use HamcrestPHPUnitIntegration;
use MediaWiki\MediaWikiServices;
use MediaWiki\Request\FauxRequest;
use MultiConfig;
use NullStatsdDataFactory;
use SpecialPageTestBase;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\Item;
use Wikibase\DataModel\Entity\Property;
use Wikibase\DataModel\Services\Statement\GuidGenerator;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\Repo\WikibaseRepo;
use WikibaseQuality\ConstraintReport\ConstraintsServices;
use WikibaseQuality\ConstraintReport\Specials\SpecialConstraintReport;
use WikibaseQuality\ConstraintReport\Tests\DefaultConfig;
use Wikimedia\Rdbms\DBUnexpectedError;

/**
 * @covers WikibaseQuality\ConstraintReport\Specials\SpecialConstraintReport
 *
 * @group WikibaseQualityConstraints
 * @group Database
 * @group medium
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class SpecialConstraintReportTest extends SpecialPageTestBase {
	use HamcrestPHPUnitIntegration;

	use DefaultConfig;

	/**
	 * Id of a item that (hopefully) does not exist.
	 */
	private const NOT_EXISTENT_ITEM_ID = 'Q2147483647';

	/**
	 * @var EntityId[]
	 */
	private static $idMap;

	/**
	 * @var bool
	 */
	private static $hasSetup;

	protected function setUp(): void {
		parent::setUp();
		MediaWikiServices::getInstance()->resetServiceForTesting( ConstraintsServices::CONSTRAINT_LOOKUP );
		$this->tablesUsed[] = 'wbqc_constraints';
		$config = new MultiConfig( [
			self::getDefaultConfig(),
			MediaWikiServices::getInstance()->getMainConfig(),
		] );
		$this->setService( 'MainConfig', $config );
	}

	protected function tearDown(): void {
		MediaWikiServices::getInstance()->resetServiceForTesting( ConstraintsServices::CONSTRAINT_LOOKUP );
		parent::tearDown();
	}

	protected function newSpecialPage() {

		return new SpecialConstraintReport(
			WikibaseRepo::getEntityLookup(),
			WikibaseRepo::getEntityTitleLookup(),
			WikibaseRepo::getEntityIdLabelFormatterFactory(),
			WikibaseRepo::getEntityIdHtmlLinkFormatterFactory(),
			WikibaseRepo::getEntityIdParser(),
			WikibaseRepo::getLanguageFallbackChainFactory(),
			ConstraintsServices::getDelegatingConstraintChecker(),
			ConstraintsServices::getViolationMessageRendererFactory(),
			self::getDefaultConfig(),
			new NullStatsdDataFactory()
		);
	}

	/**
	 * Adds temporary test data to database
	 *
	 * @throws DBUnexpectedError
	 */
	public function addDBData() {
		if ( !self::$hasSetup ) {
			$store = WikibaseRepo::getEntityStore();

			$editor = $this->getTestUser()->getUser();

			$propertyP1 = Property::newFromType( 'string' );
			$store->saveEntity( $propertyP1, 'TestEntityP1', $editor, EDIT_NEW );
			self::$idMap[ 'P1' ] = $propertyP1->getId();

			$itemQ1 = new Item();
			$store->saveEntity( $itemQ1, 'TestEntityQ1', $editor, EDIT_NEW );
			self::$idMap[ 'Q1' ] = $itemQ1->getId();

			$statementGuidGenerator = new GuidGenerator();

			$dataValue = new StringValue( 'foo' );
			$snak = new PropertyValueSnak( self::$idMap[ 'P1' ], $dataValue );
			$statement = new Statement( $snak );
			$statementGuid = $statementGuidGenerator->newGuid( self::$idMap[ 'Q1' ] );
			$statement->setGuid( $statementGuid );
			$itemQ1->getStatements()->addStatement( $statement );

			$store->saveEntity( $itemQ1, 'TestEntityQ1', $editor, EDIT_UPDATE );

			self::$hasSetup = true;
		}

		// Truncate table
		$this->db->delete( 'wbqc_constraints', '*' );

		$this->db->insert(
			'wbqc_constraints',
			[
				[
					'constraint_guid' => '1',
					'pid' => self::$idMap[ 'P1' ]->getNumericId(),
					'constraint_type_qid' =>
						self::getDefaultConfig()->get( 'WBQualityConstraintsMultiValueConstraintId' ),
					'constraint_parameters' => '{}',
				],
				[
					'constraint_guid' => '3',
					'pid' => self::$idMap[ 'P1' ]->getNumericId(),
					'constraint_type_qid' =>
						self::getDefaultConfig()->get( 'WBQualityConstraintsSingleValueConstraintId' ),
					'constraint_parameters' => '{}',
				],
			]
		);
	}

	/**
	 * @dataProvider provideRequestsAndMatchers
	 */
	public function testExecute( $subPage, array $request, $userLanguage, array $matchers ) {
		$request = new FauxRequest( $request );

		// the added item is Q1; this solves the problem that the provider is executed before the test
		$id = self::$idMap[ 'Q1' ];
		$subPage = str_replace( '$id', $id->getSerialization(), $subPage );

		// assert matchers
		list( $output, ) = $this->executeSpecialPage( $subPage, $request, $userLanguage );
		foreach ( $matchers as $key => $matcher ) {
			$this->assertThatHamcrest(
				"Failed to assert output: $key",
				$output,
				is( htmlPiece( havingChild( $matcher ) ) )
			);
		}
	}

	public static function provideRequestsAndMatchers() {
		$userLanguage = 'qqx';
		$cases = [];
		$matchers = [];

		// Empty input
		$matchers['explanationOne'] = both( withTagName( 'p' ) )
			->andAlso( havingTextContents( '(wbqc-constraintreport-explanation-part-one)' ) );

		$matchers['explanationTwo'] = both( withTagName( 'p' ) )
			->andAlso( havingTextContents( '(wbqc-constraintreport-explanation-part-two)' ) );

		$matchers['entityId'] = both(
			tagMatchingOutline( '<div class="wbqc-constraintreport-form-entity-id"/>' )
		)->andAlso(
			havingChild( tagMatchingOutline(
				'<input
					placeholder="(wbqc-constraintreport-form-entityid-placeholder)"
					name="entityid"/>'
			) )
		);

		$matchers['submit'] = both(
			tagMatchingOutline( '<button type="submit"/>' )
		)->andAlso(
			havingChild( allOf(
				withTagName( 'span' ),
				havingTextContents( '(wbqc-constraintreport-form-submit-label)' )
			) )
		);

		$cases[ 'empty' ] = [ '', [], $userLanguage, $matchers ];

		// Invalid input
		$matchers['error'] = both(
			tagMatchingOutline(
				'<p class="wbqc-constraintreport-notice wbqc-constraintreport-notice-error"/>'
			)
		)->andAlso( havingTextContents( '(wbqc-constraintreport-invalid-entity-id)' ) );

		$cases[ 'invalid input 1' ] = [ 'Qwertz', [], $userLanguage, $matchers ];
		$cases[ 'invalid input 2' ] = [ '300', [], $userLanguage, $matchers ];

		// Valid input but entity does not exist
		unset( $matchers[ 'error' ] );

		$matchers['error'] = both(
			tagMatchingOutline(
				'<p class="wbqc-constraintreport-notice wbqc-constraintreport-notice-error"/>'
			)
		)->andAlso( havingTextContents( '(wbqc-constraintreport-not-existent-entity)' ) );

		$cases[ 'valid input - not existing item' ] = [
			self::NOT_EXISTENT_ITEM_ID,
			[],
			$userLanguage,
			$matchers,
		];

		// Valid input and entity exists
		unset( $matchers[ 'error' ] );
		$matchers[ 'result for' ] = [
			'tag' => 'h3',
			'content' => '(wbqc-constraintreport-result-headline) ',
		];

		$matchers['result for'] = both(
			withTagName( 'h3' )
		)->andAlso(
			havingTextContents( containsString( '(wbqc-constraintreport-result-headline) ' ) )
		);

		$matchers['result table'] = tagMatchingOutline(
			'<table class="wikitable sortable"/>'
		);

		$matchers['column status'] = both(
			tagMatchingOutline( '<th role="columnheader button"/>' )
		)->andAlso(
			havingTextContents( '(wbqc-constraintreport-result-table-header-status)' )
		);

		$matchers['column property'] = both(
			tagMatchingOutline( '<th role="columnheader button"/>' )
		)->andAlso(
			havingTextContents( '(wbqc-constraintreport-result-table-header-property)' )
		);

		$matchers['column message'] = both(
			tagMatchingOutline( '<th role="columnheader button"/>' )
		)->andAlso(
			havingTextContents( '(wbqc-constraintreport-result-table-header-message)' )
		);

		$matchers['column constraint'] = both(
			tagMatchingOutline( '<th role="columnheader button"/>' )
		)->andAlso(
			havingTextContents( '(wbqc-constraintreport-result-table-header-constraint)' )
		);

		$matchers['value status - warning'] = both(
			tagMatchingOutline( '<span class="wbqc-status wbqc-status-warning"/>' )
		)->andAlso(
			havingChild(
				both(
					withTagName( 'label' )
				)->andAlso(
					havingTextContents( '(wbqc-constraintreport-status-warning)' )
				)
			)
		);

		$matchers['value status - compliance'] = both(
			tagMatchingOutline( '<span class="wbqc-status wbqc-status-compliance"/>' )
		)->andAlso(
			havingChild(
				both(
					withTagName( 'label' )
				)->andAlso(
					havingTextContents( '(wbqc-constraintreport-status-compliance)' )
				)
			)
		);

		$cases[ 'valid input - existing item' ] = [ '$id', [], $userLanguage, $matchers ];

		return $cases;
	}

}
