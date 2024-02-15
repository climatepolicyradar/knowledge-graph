<?php

namespace WikibaseQuality\ConstraintReport\Tests\Api;

use Article;
use HashConfig;
use IContextSource;
use MediaWiki\Request\WebResponse;
use MediaWiki\Title\Title;
use NullStatsdDataFactory;
use OutputPage;
use PHPUnit\Framework\MockObject\MockObject;
use Psr\Log\NullLogger;
use WANObjectCache;
use WebRequest;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\ItemIdParser;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\Lib\Store\Sql\WikiPageEntityMetaDataAccessor;
use Wikibase\Repo\Rdf\RdfVocabulary;
use Wikibase\Repo\WikibaseRepo;
use WikibaseQuality\ConstraintReport\Api\CachingResultsSource;
use WikibaseQuality\ConstraintReport\Api\CheckConstraintsRdf;
use WikibaseQuality\ConstraintReport\Api\ResultsCache;
use WikibaseQuality\ConstraintReport\Api\ResultsSource;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckResults;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachingMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\LoggingHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultDeserializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResultSerializer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\NullResult;

/**
 * @covers \WikibaseQuality\ConstraintReport\Api\CheckConstraintsRdf
 *
 * @group API
 * @group WikibaseAPI
 *
 * @group medium
 *
 * @license GPL-2.0-or-later
 */
class CheckConstraintsRdfTest extends \PHPUnit\Framework\TestCase {

	/**
	 * @return MockObject
	 */
	private function getOutput() {
		return $this->createMock( OutputPage::class );
	}

	/**
	 * @param string $entityId entity ID serialization
	 */
	private function getCheckResult( $entityId, $status = CheckResult::STATUS_VIOLATION ) {
		return new CheckResult(
			new MainSnakContextCursor(
				$entityId,
				'P1',
				$entityId . '$00000000-0000-0000-0000-000000000000',
				'0000000000000000000000000000000000000000'
			),
			new Constraint(
				'P1$00000000-0000-0000-0000-000000000000',
				new NumericPropertyId( 'P1' ),
				'Q12345',
				[]
			),
			$status,
			new ViolationMessage( 'wbqc-violation-message-single-value' )
		);
	}

	private function getNullResult( $entityId ) {
		return new NullResult(
			new MainSnakContextCursor(
				$entityId,
				'P1',
				$entityId . '$00000000-0000-0000-0000-000000000000',
				'0000000000000000000000000000000000000000'
			)
		);
	}

	/**
	 * @param MockObject $mockResponse
	 *
	 * @return IContextSource
	 */
	private function getContext( MockObject $mockResponse ) {
		$output = $this->getOutput();
		$context = $this->createMock( IContextSource::class );

		$mockRequest = $this->createMock( WebRequest::class );
		$mockRequest->method( 'response' )
			->willReturn( $mockResponse );
		$context->method( 'getRequest' )
			->willReturn( $mockRequest );

		$context->method( 'getOutput' )
			->willReturn( $output );

		return $context;
	}

	/**
	 * @return LoggingHelper
	 */
	private function getLoggingHelper() {
		return new LoggingHelper(
			new NullStatsdDataFactory(),
			new NullLogger(),
			new HashConfig( [
				'WBQualityConstraintsCheckDurationInfoSeconds' => 5.0,
				'WBQualityConstraintsCheckDurationWarningSeconds' => 10.0,
				'WBQualityConstraintsCheckOnEntityDurationInfoSeconds' => 15.0,
				'WBQualityConstraintsCheckOnEntityDurationWarningSeconds' => 55.0,
			] )
		);
	}

	/**
	 * @return ResultsSource
	 */
	private function getCachingResultsSource() {
		return $this->getMockBuilder( CachingResultsSource::class )
			->setConstructorArgs( [
				$this->createMock( ResultsSource::class ),
				new ResultsCache( WANObjectCache::newEmpty(), 'v2' ),
				$this->createMock( CheckResultSerializer::class ),
				$this->createMock( CheckResultDeserializer::class ),
				$this->createMock( WikiPageEntityMetaDataAccessor::class ),
				new ItemIdParser(),
				86400,
				[],
				10000,
				$this->getLoggingHelper(),
			] )
			->getMock();
	}

	private function getCheckConstraintsRdf(
		Article $article,
		$mockResponse,
		ResultsSource $cachingResultsSource = null
	): CheckConstraintsRdf {
		if ( $cachingResultsSource === null ) {
			$cachingResultsSource = $this->getCachingResultsSource();
		}

		return new CheckConstraintsRdf(
			$article,
			$this->getContext( $mockResponse ),
			$cachingResultsSource,
			WikibaseRepo::getEntityIdLookup(),
			WikibaseRepo::getRdfVocabulary()
		);
	}

	public function testShow() {
		$cachingResultsSource = $this->getCachingResultsSource();
		$cachingResultsSource->expects( $this->once() )->method( 'getStoredResults' )
			->willReturnCallback( function ( EntityId $entityId ) {
				$serialization = $entityId->getSerialization();
				return new CachedCheckResults(
					[ $this->getCheckResult( $serialization ) ],
					Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 64800 ) )
				);
			} );

		$mockResponse = $this->createMock( WebResponse::class );
		$mockResponse->expects( $this->never() )->method( 'statusHeader' );
		$mockResponse->expects( $this->once() )->method( 'header' )
			->with( 'Content-Type: text/turtle; charset=UTF-8' );
		$rdfVocabulary = WikibaseRepo::getRdfVocabulary();
		$action = $this->getCheckConstraintsRdf(
			new Article( Title::newFromText( 'Property:P1' ) ),
			$mockResponse,
			$cachingResultsSource
		);

		ob_start();
		$action->onView();
		$actualOutput = ob_get_clean();

		$wdsURI = $rdfVocabulary->getNamespaceURI( RdfVocabulary::NS_STATEMENT );
		$expectedOutput = <<<TEXT
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix s: <$wdsURI> .
@prefix wikibase: <http://wikiba.se/ontology#> .

s:P1-00000000-0000-0000-0000-000000000000 wikibase:hasViolationForConstraint s:P1-00000000-0000-0000-0000-000000000000 .

TEXT;
		$this->assertSame( $expectedOutput, $actualOutput );
	}

	public function testShow404() {
		$mockResponse = $this->createMock( WebResponse::class );
		$mockResponse->expects( $this->once() )->method( 'statusHeader' )->with( 404 );
		$action = $this->getCheckConstraintsRdf(
			new Article( Title::newFromText( 'something strange' ) ),
			$mockResponse
		);

		ob_start();
		$action->onView();
		$actualOutput = ob_get_clean();

		$expectedOutput = '';
		$this->assertSame( $expectedOutput, $actualOutput );
	}

	public function testShowNoResults() {
		$mockResponse = $this->createMock( WebResponse::class );
		$mockResponse->expects( $this->once() )->method( 'statusHeader' )->with( 204 );
		$action = $this->getCheckConstraintsRdf(
			new Article( Title::newFromText( 'Item:Q1' ) ),
			$mockResponse
		);

		ob_start();
		$action->onView();
		$actualOutput = ob_get_clean();

		$expectedOutput = '';
		$this->assertSame( $expectedOutput, $actualOutput );
	}

	public function testShowNoResultsWithNull() {
		$cachingResultsSource = $this->getCachingResultsSource();
		$cachingResultsSource->expects( $this->once() )->method( 'getStoredResults' )
			->willReturnCallback( function ( EntityId $entityId ) {
				$serialization = $entityId->getSerialization();
				return new CachedCheckResults(
					[
						$this->getNullResult( $serialization ),
						$this->getCheckResult( $serialization, CheckResult::STATUS_BAD_PARAMETERS ),
					],
					Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 64800 ) )
				);
			} );

		$mockResponse = $this->createMock( WebResponse::class );
		$mockResponse->expects( $this->once() )->method( 'statusHeader' )->with( 204 );
		$action = $this->getCheckConstraintsRdf(
			new Article( Title::newFromText( 'Property:P1' ) ),
			$mockResponse,
			$cachingResultsSource
		);

		ob_start();
		$action->onView();
		$actualOutput = ob_get_clean();

		$expectedOutput = '';
		$this->assertSame( $expectedOutput, $actualOutput );
	}

}
