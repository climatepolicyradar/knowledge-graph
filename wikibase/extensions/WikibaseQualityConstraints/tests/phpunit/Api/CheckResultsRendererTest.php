<?php

namespace WikibaseQuality\ConstraintReport\Tests\Api;

use DataValues\MonolingualTextValue;
use DataValues\MultilingualTextValue;
use MediaWiki\MediaWikiServices;
use MediaWiki\Title\Title;
use MockMessageLocalizer;
use ValueFormatters\ValueFormatter;
use Wikibase\DataModel\Entity\EntityId;
use Wikibase\DataModel\Entity\Item;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\EntityId\PlainEntityIdFormatter;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Statement\Statement;
use Wikibase\Lib\Store\EntityTitleLookup;
use Wikibase\Lib\TermLanguageFallbackChain;
use WikibaseQuality\ConstraintReport\Api\CheckResultsRenderer;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachedCheckResults;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\CachingMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\DependencyMetadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Cache\Metadata;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContextCursor;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\QualifierContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\NullResult;
use WikibaseQuality\ConstraintReport\Tests\DefaultConfig;
use WikibaseQuality\ConstraintReport\Tests\Fake\AppendingContextCursor;

/**
 * @covers WikibaseQuality\ConstraintReport\Api\CheckResultsRenderer
 *
 * @license GPL-2.0-or-later
 */
class CheckResultsRendererTest extends \PHPUnit\Framework\TestCase {

	use DefaultConfig;

	private function getResultsRenderer() {
		$entityIdFormatter = new PlainEntityIdFormatter();
		$entityTitleLookup = $this->createMock( EntityTitleLookup::class );
		$entityTitleLookup->method( 'getTitleForId' )
			->willReturnCallback( function ( EntityId $id ) {
				$title = $this->createMock( Title::class );
				$title->method( 'getFullUrl' )
					->willReturn( 'http://wiki.test/' . $id->getSerialization() );
				$title->method( 'getTalkPageIfDefined' )
					->willReturnCallback( function () use ( $id ) {
						$title = $this->createMock( Title::class );
						$title->method( 'getFullUrl' )
							->willReturn( 'http://wiki.test/Talk:' . $id->getSerialization() );
						return $title;
					} );
				return $title;
			} );
		$valueFormatter = $this->createMock( ValueFormatter::class );

		$userLanguageCode = 'de-at';
		$languageFallbackChain = $this->createConfiguredMock( TermLanguageFallbackChain::class, [
			'getFetchLanguageCodes' => [ 'de-at', 'de', 'en' ],
		] );

		return new CheckResultsRenderer(
			$entityTitleLookup,
			$entityIdFormatter,
			$languageFallbackChain,
			new ViolationMessageRenderer(
				$entityIdFormatter,
				$valueFormatter,
				MediaWikiServices::getInstance()->getLanguageNameUtils(),
				$userLanguageCode,
				$languageFallbackChain,
				new MockMessageLocalizer(),
				self::getDefaultConfig()
			)
		);
	}

	public function testRender_EmptyArrayWithMetadata() {
		$q1 = new ItemId( 'Q1' );
		$checkResults = new CachedCheckResults(
			[],
			Metadata::merge( [
				Metadata::ofCachingMetadata( CachingMetadata::ofMaximumAgeInSeconds( 300 ) ),
				Metadata::ofDependencyMetadata( DependencyMetadata::ofEntityId( $q1 ) ),
			] )
		);

		$result = $this->getResultsRenderer()->render( $checkResults );
		$metadata = $result->getMetadata();

		$this->assertSame( [], $result->getArray() );
		$this->assertSame( 300, $metadata->getCachingMetadata()->getMaximumAgeInSeconds() );
		$this->assertSame( [ $q1 ], $metadata->getDependencyMetadata()->getEntityIds() );
	}

	public function testRender_TwoResults() {
		$constraint = new Constraint(
			'P31$26b6340b-5257-4a9f-94d6-f2e01b539484',
			new NumericPropertyId( 'P31' ),
			'Q21510857',
			[]
		);
		$q1 = new ItemId( 'Q1' );
		$q4 = new ItemId( 'Q4' );

		$checkResult1 = ( new CheckResult(
			new MainSnakContextCursor(
				'Q1',
				'P31',
				'Q1$8983b0ea-4a9c-0902-c0db-785db33f767c',
				'a35ee6b06a0f0e78614b517e4b72029b535479c0'
			),
			$constraint,
			CheckResult::STATUS_VIOLATION,
			new ViolationMessage( 'wbqc-violation-message-multi-value' )
		) )->withMetadata(
			Metadata::ofDependencyMetadata( DependencyMetadata::ofEntityId( $q1 ) )
		);
		$checkResult2 = ( new CheckResult(
			new MainSnakContextCursor(
				'Q4',
				'P31',
				'Q4$9d34c155-455a-39da-1940-fde1f4f00434',
				'8d7709b2902c132ce035a10367b7b6044e6bbc07'
			),
			$constraint,
			CheckResult::STATUS_COMPLIANCE
		) )->withMetadata(
			Metadata::ofDependencyMetadata( DependencyMetadata::ofEntityId( $q4 ) )
		);

		$checkResults = new CachedCheckResults(
			[ $checkResult1, $checkResult2 ],
			Metadata::merge( [ $checkResult1->getMetadata(), $checkResult2->getMetadata() ] )
		);

		$result = $this->getResultsRenderer()->render( $checkResults );

		$this->assertSame( [ 'Q1', 'Q4' ], array_keys( $result->getArray() ) );
		foreach ( $result->getArray() as $resultByQ ) {
			$this->assertSame( [ 'P31' ], array_keys( $resultByQ['claims'] ) );
			$this->assertCount( 1, $resultByQ['claims']['P31'] );
			$this->assertCount( 1, $resultByQ['claims']['P31'][0]['mainsnak']['results'] );
		}
		$this->assertSame(
			[ $q1, $q4 ],
			$result->getMetadata()->getDependencyMetadata()->getEntityIds()
		);
	}

	public function testCheckResultToArray_NullResult() {
		$checkResult = new NullResult( new AppendingContextCursor() );

		$result = $this->getResultsRenderer()->checkResultToArray( $checkResult );

		$this->assertNull( $result );
	}

	public function testCheckResultToArray_Constraint() {
		$checkResult = new CheckResult(
			new MainSnakContext(
				new Item( new ItemId( 'Q1' ) ),
				new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) )
			),
			new Constraint(
				'P1$31d77e02-e1bd-423e-811e-7f6dd5da0b90',
				new NumericPropertyId( 'P1' ),
				'Q1',
				[]
			)
		);

		$result = $this->getResultsRenderer()->checkResultToArray( $checkResult );
		$constraint = $result['constraint'];

		$this->assertSame( $checkResult->getConstraintId(), $constraint['id'] );
		$this->assertSame( 'Q1', $constraint['type'] );
		$this->assertSame( 'Q1', $constraint['typeLabel'] );
		$this->assertSame( 'http://wiki.test/P1#P1$31d77e02-e1bd-423e-811e-7f6dd5da0b90', $constraint['link'] );
		$this->assertSame( 'http://wiki.test/Talk:P1', $constraint['discussLink'] );
	}

	/** @dataProvider provideConstraintClarifications */
	public function testCheckResultToArray_Result(
		MultilingualTextValue $constraintClarification,
		?string $expectedClarification
	): void {
		$checkResult = new CheckResult(
			new MainSnakContext(
				new Item( new ItemId( 'Q1' ) ),
				new Statement(
					new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ),
					null,
					null,
					'Q1$1deb7c9e-8de4-4bc2-baab-add9d4f538c3'
				)
			),
			new Constraint(
				'P1$31d77e02-e1bd-423e-811e-7f6dd5da0b90',
				new NumericPropertyId( 'P1' ),
				'Q1',
				[]
			),
			'status',
			( new ViolationMessage( 'wbqc-violation-message-no-qualifiers' ) )
				->withEntityId( new NumericPropertyId( 'P1' ) )
		);
		$checkResult->setConstraintClarification( $constraintClarification );

		$result = $this->getResultsRenderer()->checkResultToArray( $checkResult );

		$this->assertSame( 'status', $result['status'] );
		$this->assertSame( 'P1', $result['property'] );
		$this->assertSame( '(wbqc-violation-message-no-qualifiers: P1)', $result['message-html'] );
		if ( $expectedClarification !== null ) {
			$this->assertSame( $expectedClarification, $result['constraint-clarification'] );
		} else {
			$this->assertArrayNotHasKey( 'constraint-clarification', $result );
		}
		$this->assertSame( 'Q1$1deb7c9e-8de4-4bc2-baab-add9d4f538c3', $result['claim'] );
		$this->assertArrayNotHasKey( 'cached', $result );
	}

	public static function provideConstraintClarifications(): iterable {
		yield 'exact match' => [
			'constraintClarification' => new MultilingualTextValue( [
				new MonolingualTextValue( 'de-at', 'de-at clarification' ),
			] ),
			'expectedClarification' => 'de-at clarification',
		];
		yield 'with language fallback' => [
			'constraintClarification' => new MultilingualTextValue( [
				new MonolingualTextValue( 'pt', 'pt clarification' ),
				new MonolingualTextValue( 'de', 'de clarification' ),
				new MonolingualTextValue( 'en', 'en clarification' ),
			] ),
			'expectedClarification' => 'de clarification',
		];
		yield 'no match in fallback chain' => [
			'constraintClarification' => new MultilingualTextValue( [
				new MonolingualTextValue( 'pt', 'pt clarification' ),
			] ),
			'expectedClarification' => null,
		];
		yield 'no clarification' => [
			'constraintClarification' => new MultilingualTextValue( [] ),
			'expectedClarification' => null,
		];
	}

	public function testCheckResultToArray_Qualifier() {
		$checkResult = new CheckResult(
			new QualifierContext(
				new Item( new ItemId( 'Q1' ) ),
				new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P2' ) ) ),
				new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) )
			),
			new Constraint(
				'P1$31d77e02-e1bd-423e-811e-7f6dd5da0b90',
				new NumericPropertyId( 'P1' ),
				'Q1',
				[]
			)
		);
		$checkResult->withMetadata( Metadata::ofCachingMetadata(
			CachingMetadata::ofMaximumAgeInSeconds( 10 ) ) );

		$result = $this->getResultsRenderer()->checkResultToArray( $checkResult );

		$this->assertSame( 'P1', $result['property'] );
		$this->assertArrayNotHasKey( 'claim', $result );
	}

	public function testCheckResultToArray_Cached() {
		$checkResult = new CheckResult(
			new MainSnakContext(
				new Item( new ItemId( 'Q1' ) ),
				new Statement( new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) ) )
			),
			new Constraint(
				'P1$31d77e02-e1bd-423e-811e-7f6dd5da0b90',
				new NumericPropertyId( 'P1' ),
				'Q1',
				[]
			)
		);
		$checkResult->withMetadata( Metadata::ofCachingMetadata(
			CachingMetadata::ofMaximumAgeInSeconds( 10 ) ) );

		$result = $this->getResultsRenderer()->checkResultToArray( $checkResult );

		$this->assertSame( [ 'maximumAgeInSeconds' => 10 ], $result['cached'] );
	}

}
