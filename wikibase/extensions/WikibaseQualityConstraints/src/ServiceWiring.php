<?php

namespace WikibaseQuality\ConstraintReport;

use MediaWiki\Logger\LoggerFactory;
use MediaWiki\MediaWikiServices;
use MediaWiki\WikiMap\WikiMap;
use ObjectCache;
use RuntimeException;
use Wikibase\DataModel\Entity\Property;
use Wikibase\Repo\WikibaseRepo;
use WikibaseQuality\ConstraintReport\Api\CachingResultsSource;
use WikibaseQuality\ConstraintReport\Api\CheckingResultsSource;
use WikibaseQuality\ConstraintReport\Api\ExpiryLock;
use WikibaseQuality\ConstraintReport\Api\ResultsCache;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursorDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\ContextCursorSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\DelegatingConstraintChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConnectionCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterParser;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\DummySparqlHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\LoggingHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\RangeCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\SparqlHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\TypeCheckerHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRendererFactory;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultSerializer;

return [
	ConstraintsServices::EXPIRY_LOCK => static function ( MediaWikiServices $services ) {
		return new ExpiryLock( ObjectCache::getInstance( CACHE_ANYTHING ) );
	},

	ConstraintsServices::LOGGING_HELPER => static function ( MediaWikiServices $services ) {
		return new LoggingHelper(
			$services->getStatsdDataFactory(),
			LoggerFactory::getInstance( 'WikibaseQualityConstraints' ),
			$services->getMainConfig()
		);
	},

	ConstraintsServices::CONSTRAINT_STORE => static function ( MediaWikiServices $services ) {
		$sourceDefinitions = WikibaseRepo::getEntitySourceDefinitions( $services );
		$propertySource = $sourceDefinitions->getDatabaseSourceForEntityType( Property::ENTITY_TYPE );
		if ( $propertySource === null ) {
			throw new RuntimeException( 'Can\'t get a ConstraintStore for properties not stored in a database.' );
		}

		$localEntitySourceName = WikibaseRepo::getLocalEntitySource( $services )->getSourceName();
		if ( $propertySource->getSourceName() !== $localEntitySourceName ) {
			throw new RuntimeException( 'Can\'t get a ConstraintStore for a non local entity source.' );
		}

		$dbName = $propertySource->getDatabaseName();
		return new ConstraintRepositoryStore(
			$services->getDBLoadBalancerFactory()->getMainLB( $dbName ),
			$dbName
		);
	},

	ConstraintsServices::CONSTRAINT_LOOKUP => static function ( MediaWikiServices $services ) {
		$sourceDefinitions = WikibaseRepo::getEntitySourceDefinitions( $services );
		$propertySource = $sourceDefinitions->getDatabaseSourceForEntityType( Property::ENTITY_TYPE );
		if ( $propertySource === null ) {
			throw new RuntimeException( 'Can\'t get a ConstraintStore for properties not stored in a database.' );
		}

		$dbName = $propertySource->getDatabaseName();
		$rawLookup = new ConstraintRepositoryLookup(
			$services->getDBLoadBalancerFactory()->getMainLB( $dbName ),
			$dbName
		);
		return new CachingConstraintLookup( $rawLookup );
	},

	ConstraintsServices::CHECK_RESULT_SERIALIZER => static function ( MediaWikiServices $services ) {
		return new CheckResultSerializer(
			new ConstraintSerializer(
				false // constraint parameters are not exposed
			),
			new ContextCursorSerializer(),
			new ViolationMessageSerializer(),
			false // unnecessary to serialize individual result dependencies
		);
	},

	ConstraintsServices::CHECK_RESULT_DESERIALIZER => static function ( MediaWikiServices $services ) {
		$entityIdParser = WikibaseRepo::getEntityIdParser( $services );
		$dataValueFactory = WikibaseRepo::getDataValueFactory( $services );

		return new CheckResultDeserializer(
			new ConstraintDeserializer(),
			new ContextCursorDeserializer(),
			new ViolationMessageDeserializer(
				$entityIdParser,
				$dataValueFactory
			),
			$entityIdParser
		);
	},

	ConstraintsServices::VIOLATION_MESSAGE_SERIALIZER => static function ( MediaWikiServices $services ) {
		return new ViolationMessageSerializer();
	},

	ConstraintsServices::VIOLATION_MESSAGE_DESERIALIZER => static function ( MediaWikiServices $services ) {
		$entityIdParser = WikibaseRepo::getEntityIdParser( $services );
		$dataValueFactory = WikibaseRepo::getDataValueFactory( $services );

		return new ViolationMessageDeserializer(
			$entityIdParser,
			$dataValueFactory
		);
	},

	ConstraintsServices::CONSTRAINT_PARAMETER_PARSER => static function ( MediaWikiServices $services ) {
		$deserializerFactory = WikibaseRepo::getBaseDataModelDeserializerFactory( $services );
		$entitySourceDefinitions = WikibaseRepo::getEntitySourceDefinitions( $services );

		return new ConstraintParameterParser(
			$services->getMainConfig(),
			$deserializerFactory,
			$entitySourceDefinitions->getDatabaseSourceForEntityType( 'item' )->getConceptBaseUri()
		);
	},

	ConstraintsServices::CONNECTION_CHECKER_HELPER => static function ( MediaWikiServices $services ) {
		return new ConnectionCheckerHelper();
	},

	ConstraintsServices::RANGE_CHECKER_HELPER => static function ( MediaWikiServices $services ) {
		return new RangeCheckerHelper(
			$services->getMainConfig(),
			WikibaseRepo::getUnitConverter( $services )
		);
	},

	ConstraintsServices::SPARQL_HELPER => static function ( MediaWikiServices $services ) {
		$endpoint = $services->getMainConfig()->get( 'WBQualityConstraintsSparqlEndpoint' );
		if ( $endpoint === '' ) {
			return new DummySparqlHelper();
		}

		$rdfVocabulary = WikibaseRepo::getRdfVocabulary( $services );
		$entityIdParser = WikibaseRepo::getEntityIdParser( $services );
		$propertyDataTypeLookup = WikibaseRepo::getPropertyDataTypeLookup( $services );

		return new SparqlHelper(
			$services->getMainConfig(),
			$rdfVocabulary,
			$entityIdParser,
			$propertyDataTypeLookup,
			$services->getMainWANObjectCache(),
			ConstraintsServices::getViolationMessageSerializer( $services ),
			ConstraintsServices::getViolationMessageDeserializer( $services ),
			$services->getStatsdDataFactory(),
			ConstraintsServices::getExpiryLock( $services ),
			ConstraintsServices::getLoggingHelper( $services ),
			WikiMap::getCurrentWikiId() . ' WikibaseQualityConstraints ' . $services->getHttpRequestFactory()->getUserAgent(),
			$services->getHttpRequestFactory()
		);
	},

	ConstraintsServices::TYPE_CHECKER_HELPER => static function ( MediaWikiServices $services ) {
		return new TypeCheckerHelper(
			WikibaseServices::getEntityLookup( $services ),
			$services->getMainConfig(),
			ConstraintsServices::getSparqlHelper( $services ),
			$services->getStatsdDataFactory()
		);
	},

	ConstraintsServices::DELEGATING_CONSTRAINT_CHECKER => static function ( MediaWikiServices $services ) {
		$statementGuidParser = WikibaseRepo::getStatementGuidParser( $services );

		$config = $services->getMainConfig();
		$checkerMap = [
			$config->get( 'WBQualityConstraintsConflictsWithConstraintId' )
				=> ConstraintCheckerServices::getConflictsWithChecker( $services ),
			$config->get( 'WBQualityConstraintsItemRequiresClaimConstraintId' )
				=> ConstraintCheckerServices::getItemChecker( $services ),
			$config->get( 'WBQualityConstraintsValueRequiresClaimConstraintId' )
				=> ConstraintCheckerServices::getTargetRequiredClaimChecker( $services ),
			$config->get( 'WBQualityConstraintsSymmetricConstraintId' )
				=> ConstraintCheckerServices::getSymmetricChecker( $services ),
			$config->get( 'WBQualityConstraintsInverseConstraintId' )
				=> ConstraintCheckerServices::getInverseChecker( $services ),
			$config->get( 'WBQualityConstraintsUsedAsQualifierConstraintId' )
				=> ConstraintCheckerServices::getQualifierChecker( $services ),
			$config->get( 'WBQualityConstraintsAllowedQualifiersConstraintId' )
				=> ConstraintCheckerServices::getQualifiersChecker( $services ),
			$config->get( 'WBQualityConstraintsMandatoryQualifierConstraintId' )
				=> ConstraintCheckerServices::getMandatoryQualifiersChecker( $services ),
			$config->get( 'WBQualityConstraintsRangeConstraintId' )
				=> ConstraintCheckerServices::getRangeChecker( $services ),
			$config->get( 'WBQualityConstraintsDifferenceWithinRangeConstraintId' )
				=> ConstraintCheckerServices::getDiffWithinRangeChecker( $services ),
			$config->get( 'WBQualityConstraintsTypeConstraintId' )
				=> ConstraintCheckerServices::getTypeChecker( $services ),
			$config->get( 'WBQualityConstraintsValueTypeConstraintId' )
				=> ConstraintCheckerServices::getValueTypeChecker( $services ),
			$config->get( 'WBQualityConstraintsSingleValueConstraintId' )
				=> ConstraintCheckerServices::getSingleValueChecker( $services ),
			$config->get( 'WBQualityConstraintsMultiValueConstraintId' )
				=> ConstraintCheckerServices::getMultiValueChecker( $services ),
			$config->get( 'WBQualityConstraintsDistinctValuesConstraintId' )
				=> ConstraintCheckerServices::getUniqueValueChecker( $services ),
			$config->get( 'WBQualityConstraintsFormatConstraintId' )
				=> ConstraintCheckerServices::getFormatChecker( $services ),
			$config->get( 'WBQualityConstraintsCommonsLinkConstraintId' )
				=> ConstraintCheckerServices::getCommonsLinkChecker( $services ),
			$config->get( 'WBQualityConstraintsOneOfConstraintId' )
				=> ConstraintCheckerServices::getOneOfChecker( $services ),
			$config->get( 'WBQualityConstraintsUsedForValuesOnlyConstraintId' )
				=> ConstraintCheckerServices::getValueOnlyChecker( $services ),
			$config->get( 'WBQualityConstraintsUsedAsReferenceConstraintId' )
				=> ConstraintCheckerServices::getReferenceChecker( $services ),
			$config->get( 'WBQualityConstraintsNoBoundsConstraintId' )
				=> ConstraintCheckerServices::getNoBoundsChecker( $services ),
			$config->get( 'WBQualityConstraintsAllowedUnitsConstraintId' )
				=> ConstraintCheckerServices::getAllowedUnitsChecker( $services ),
			$config->get( 'WBQualityConstraintsSingleBestValueConstraintId' )
				=> ConstraintCheckerServices::getSingleBestValueChecker( $services ),
			$config->get( 'WBQualityConstraintsAllowedEntityTypesConstraintId' )
				=> ConstraintCheckerServices::getEntityTypeChecker( $services ),
			$config->get( 'WBQualityConstraintsNoneOfConstraintId' )
				=> ConstraintCheckerServices::getNoneOfChecker( $services ),
			$config->get( 'WBQualityConstraintsIntegerConstraintId' )
				=> ConstraintCheckerServices::getIntegerChecker( $services ),
			$config->get( 'WBQualityConstraintsCitationNeededConstraintId' )
				=> ConstraintCheckerServices::getCitationNeededChecker( $services ),
			$config->get( 'WBQualityConstraintsPropertyScopeConstraintId' )
				=> ConstraintCheckerServices::getPropertyScopeChecker( $services ),
			$config->get( 'WBQualityConstraintsContemporaryConstraintId' )
				=> ConstraintCheckerServices::getContemporaryChecker( $services ),
			$config->get( 'WBQualityConstraintsLexemeLanguageConstraintId' )
				=> ConstraintCheckerServices::getLexemeLanguageChecker( $services ),
			$config->get( 'WBQualityConstraintsLabelInLanguageConstraintId' )
				=> ConstraintCheckerServices::getLabelInLanguageChecker( $services ),
		];

		return new DelegatingConstraintChecker(
			WikibaseServices::getEntityLookup( $services ),
			$checkerMap,
			ConstraintsServices::getConstraintLookup( $services ),
			ConstraintsServices::getConstraintParameterParser( $services ),
			$statementGuidParser,
			ConstraintsServices::getLoggingHelper( $services ),
			$config->get( 'WBQualityConstraintsCheckQualifiers' ),
			$config->get( 'WBQualityConstraintsCheckReferences' ),
			$config->get( 'WBQualityConstraintsPropertiesWithViolatingQualifiers' )
		);
	},

	ConstraintsServices::RESULTS_SOURCE => static function ( MediaWikiServices $services ) {
		$config = $services->getMainConfig();
		$resultsSource = new CheckingResultsSource(
			ConstraintsServices::getDelegatingConstraintChecker( $services )
		);

		$cacheCheckConstraintsResults = false;

		if ( $config->get( 'WBQualityConstraintsCacheCheckConstraintsResults' ) ) {
			$cacheCheckConstraintsResults = true;
			// check that we can use getLocalRepoWikiPageMetaDataAccessor()
			// TODO we should always be able to cache constraint check results (T244726)
			$entitySources = WikibaseRepo::getEntitySourceDefinitions( $services )->getSources();
			$localEntitySourceName = WikibaseRepo::getLocalEntitySource( $services )->getSourceName();

			foreach ( $entitySources as $entitySource ) {
				if ( $entitySource->getSourceName() !== $localEntitySourceName ) {
					LoggerFactory::getInstance( 'WikibaseQualityConstraints' )->warning(
						'Cannot cache constraint check results for non-local source: ' .
						$entitySource->getSourceName()
					);
					$cacheCheckConstraintsResults = false;
					break;
				}
			}
		}

		if ( $cacheCheckConstraintsResults ) {
			$possiblyStaleConstraintTypes = [
				$config->get( 'WBQualityConstraintsCommonsLinkConstraintId' ),
				$config->get( 'WBQualityConstraintsTypeConstraintId' ),
				$config->get( 'WBQualityConstraintsValueTypeConstraintId' ),
				$config->get( 'WBQualityConstraintsDistinctValuesConstraintId' ),
			];
			$entityIdParser = WikibaseRepo::getEntityIdParser( $services );
			$wikiPageEntityMetaDataAccessor = WikibaseRepo::getLocalRepoWikiPageMetaDataAccessor(
				$services );

			$resultsSource = new CachingResultsSource(
				$resultsSource,
				ResultsCache::getDefaultInstance(),
				ConstraintsServices::getCheckResultSerializer( $services ),
				ConstraintsServices::getCheckResultDeserializer( $services ),
				$wikiPageEntityMetaDataAccessor,
				$entityIdParser,
				$config->get( 'WBQualityConstraintsCacheCheckConstraintsTTLSeconds' ),
				$possiblyStaleConstraintTypes,
				$config->get( 'WBQualityConstraintsCacheCheckConstraintsMaximumRevisionIds' ),
				ConstraintsServices::getLoggingHelper( $services )
			);
		}

		return $resultsSource;
	},

	ConstraintsServices::VIOLATION_MESSAGE_RENDERER_FACTORY => static function ( MediaWikiServices $services ) {
		return new ViolationMessageRendererFactory(
			$services->getMainConfig(),
			$services->getLanguageNameUtils(),
			WikibaseRepo::getEntityIdHtmlLinkFormatterFactory( $services ),
			WikibaseRepo::getValueFormatterFactory( $services )
		);
	},
];
