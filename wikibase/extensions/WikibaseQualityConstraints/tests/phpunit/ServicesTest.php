<?php

namespace WikibaseQuality\ConstraintReport\Tests;

use HashConfig;
use MediaWiki\MediaWikiServices;
use MediaWikiIntegrationTestCase;
use Wikibase\DataModel\Services\Lookup\EntityLookup;
use Wikibase\DataModel\Services\Lookup\PropertyDataTypeLookup;
use WikibaseQuality\ConstraintReport\Api\ExpiryLock;
use WikibaseQuality\ConstraintReport\Api\ResultsSource;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\AllowedUnitsChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\CitationNeededChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\CommonsLinkChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ConflictsWithChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ContemporaryChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\DiffWithinRangeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\EntityTypeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\FormatChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\IntegerChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\InverseChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ItemChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\MandatoryQualifiersChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\MultiValueChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\NoBoundsChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\NoneOfChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\OneOfChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\PropertyScopeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\QualifierChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\QualifiersChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\RangeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ReferenceChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\SingleBestValueChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\SingleValueChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\SymmetricChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\TargetRequiredClaimChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\TypeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\UniqueValueChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ValueOnlyChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\ValueTypeChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\DelegatingConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConnectionCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\LoggingHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\RangeCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\SparqlHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\TypeCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheckerServices;
use WikibaseQuality\ConstraintReport\ConstraintLookup;
use WikibaseQuality\ConstraintReport\ConstraintsServices;
use WikibaseQuality\ConstraintReport\ConstraintStore;
use WikibaseQuality\ConstraintReport\WikibaseServices;

/**
 * @covers \WikibaseQuality\ConstraintReport\ConstraintsServices
 *
 * @group WikibaseQualityConstraints
 *
 * @license GPL-2.0-or-later
 */
class ServicesTest extends MediaWikiIntegrationTestCase {

	public static function provideConstraintsServiceClasses() {
		return [
			[ ExpiryLock::class ],
			[ LoggingHelper::class ],
			[ ConstraintStore::class ],
			[ ConstraintLookup::class ],
			[ CheckResultSerializer::class ],
			[ CheckResultDeserializer::class ],
			[ ViolationMessageSerializer::class ],
			[ ViolationMessageDeserializer::class ],
			[ ConstraintParameterParser::class ],
			[ ConnectionCheckerHelper::class ],
			[ RangeCheckerHelper::class ],
			[ SparqlHelper::class, [
				'config' => new HashConfig( [ 'WBQualityConstraintsSparqlEndpoint' => 'http://f.oo/sparql' ] ) ],
			],
			[ SparqlHelper::class, [ 'config' => new HashConfig( [ 'WBQualityConstraintsSparqlEndpoint' => '' ] ) ] ],
			[ TypeCheckerHelper::class ],
			[ DelegatingConstraintChecker::class ],
			[ ResultsSource::class, [
				'config' => new HashConfig( [ 'WBQualityConstraintsCacheCheckConstraintsResults' => true ] ) ],
			],
			[ ResultsSource::class, [
				'config' => new HashConfig( [ 'WBQualityConstraintsCacheCheckConstraintsResults' => false ] ) ],
			],
		];
	}

	public static function provideConstraintCheckerServiceClasses() {
		return [
			[ ConflictsWithChecker::class ],
			[ ItemChecker::class ],
			[ TargetRequiredClaimChecker::class ],
			[ SymmetricChecker::class ],
			[ InverseChecker::class ],
			[ QualifierChecker::class ],
			[ QualifiersChecker::class ],
			[ MandatoryQualifiersChecker::class ],
			[ RangeChecker::class ],
			[ DiffWithinRangeChecker::class ],
			[ TypeChecker::class ],
			[ ValueTypeChecker::class ],
			[ SingleValueChecker::class ],
			[ MultiValueChecker::class ],
			[ UniqueValueChecker::class ],
			[ FormatChecker::class ],
			[ CommonsLinkChecker::class ],
			[ OneOfChecker::class ],
			[ ValueOnlyChecker::class ],
			[ ReferenceChecker::class ],
			[ NoBoundsChecker::class ],
			[ AllowedUnitsChecker::class ],
			[ SingleBestValueChecker::class ],
			[ EntityTypeChecker::class ],
			[ NoneOfChecker::class ],
			[ IntegerChecker::class ],
			[ CitationNeededChecker::class ],
			[ PropertyScopeChecker::class ],
			[ ContemporaryChecker::class ],
		];
	}

	public static function provideWikibaseServiceClasses() {
		return [
			[ EntityLookup::class ],
			[ 'EntityLookupWithoutCache', [ 'expectedClass' => EntityLookup::class ] ],
			[ PropertyDataTypeLookup::class ],
		];
	}

	public static function provideAllServiceClasses() {
		yield from self::provideConstraintsServiceClasses();
		yield from self::provideConstraintCheckerServiceClasses();
		yield from self::provideWikibaseServiceClasses();
	}

	/**
	 * @dataProvider provideAllServiceClasses
	 */
	public function testServiceWiring( $serviceClass, array $options = [] ) {
		$options += [ 'config' => null, 'expectedClass' => $serviceClass ];
		$this->overrideMwServices( $options['config'] );
		$serviceClassParts = explode( '\\', $serviceClass );
		$serviceName = 'WBQC_' . end( $serviceClassParts );

		$service = MediaWikiServices::getInstance()->getService( $serviceName );

		$this->assertInstanceOf( $options['expectedClass'], $service );
	}

	/**
	 * @dataProvider provideConstraintsServiceClasses
	 */
	public function testConstraintsServices( $serviceClass, array $options = [] ) {
		$options += [ 'config' => null, 'expectedClass' => $serviceClass ];
		$this->overrideMwServices( $options['config'] );
		$serviceClassParts = explode( '\\', $serviceClass );
		$getterName = 'get' . end( $serviceClassParts );

		$service = ConstraintsServices::$getterName();

		$this->assertInstanceOf( $options['expectedClass'], $service );
	}

	/**
	 * @dataProvider provideConstraintCheckerServiceClasses
	 */
	public function testConstraintCheckerServices( $serviceClass, array $options = [] ) {
		$options += [ 'config' => null, 'expectedClass' => $serviceClass ];
		$this->overrideMwServices( $options['config'] );
		$serviceClassParts = explode( '\\', $serviceClass );
		$getterName = 'get' . end( $serviceClassParts );

		$service = ConstraintCheckerServices::$getterName();

		$this->assertInstanceOf( $options['expectedClass'], $service );
	}

	/**
	 * @dataProvider provideWikibaseServiceClasses
	 */
	public function testWikibaseServices( $serviceClass, array $options = [] ) {
		$options += [ 'config' => null, 'expectedClass' => $serviceClass ];
		$this->overrideMwServices( $options['config'] );
		$serviceClassParts = explode( '\\', $serviceClass );
		$getterName = 'get' . end( $serviceClassParts );

		$service = WikibaseServices::$getterName();

		$this->assertInstanceOf( $options['expectedClass'], $service );
	}

	public function testConstraintStoreThrowsExceptionForNonLocalEntitySource() {
		$this->mergeMwGlobalArrayValue(
			'wgWBRepoSettings',
			[ 'localEntitySourceName' => 'LocalItemOnlySource' ]
		);
		$this->mergeMwGlobalArrayValue(
			'wgWBRepoSettings',
			[ 'entitySources' => [
				'LocalItemOnlySource' => $this->getASourceDefinition( [ 'item' => 23 ] ),
				'NonWritablePropertySource' => $this->getASourceDefinition( [ 'property' => 9 ] ),
			] ]
		);

		$this->expectException( \RuntimeException::class );
		ConstraintsServices::getConstraintStore();
	}

	private function getASourceDefinition( $entityNamespaces ) {
		return [
			'entityNamespaces' => $entityNamespaces,
			'repoDatabase' => 'adb',
			'baseUri' => 'http://bla/',
			'rdfNodeNamespacePrefix' => 'h',
			'rdfPredicateNamespacePrefix' => 'f',
			'interwikiPrefix' => 'wiki',
		];
	}

}
