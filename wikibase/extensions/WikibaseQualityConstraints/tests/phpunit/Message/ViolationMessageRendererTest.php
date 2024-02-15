<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\Tests\Message;

use Config;
use DataValues\StringValue;
use HashConfig;
use InvalidArgumentException;
use MediaWiki\Languages\LanguageNameUtils;
use Message;
use MockMessageLocalizer;
use ValueFormatters\StringFormatter;
use ValueFormatters\ValueFormatter;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Services\EntityId\EntityIdFormatter;
use Wikibase\DataModel\Services\EntityId\PlainEntityIdFormatter;
use Wikibase\Lib\Formatters\UnDeserializableValueFormatter;
use Wikibase\Lib\TermLanguageFallbackChain;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\Context;
use WikibaseQuality\ConstraintReport\ConstraintCheck\ItemIdSnakValue;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessage;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer;
use WikibaseQuality\ConstraintReport\Role;
use Wikimedia\TestingAccessWrapper;

/**
 * @group WikibaseQualityConstraints
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
class ViolationMessageRendererTest extends \PHPUnit\Framework\TestCase {

	/**
	 * Create a new ViolationMessageRenderer
	 * with some constructor arguments defaulting to a simple base implementation.
	 */
	private function newViolationMessageRenderer(
		EntityIdFormatter $entityIdFormatter = null,
		ValueFormatter $dataValueFormatter = null,
		Config $config = null,
		int $maxListLength = 10
	): ViolationMessageRenderer {
		if ( $entityIdFormatter === null ) {
			$entityIdFormatter = new PlainEntityIdFormatter();
		}
		if ( $dataValueFormatter === null ) {
			$dataValueFormatter = new UnDeserializableValueFormatter();
		}
		$userLanguageCode = 'en';
		$languageNameUtils = $this->createMock( LanguageNameUtils::class );
		$languageNameUtils->method( 'getLanguageName' )
			->with( 'pt', $userLanguageCode )
			->willReturn( 'Portuguese' );
		$languageFallbackChain = $this->createConfiguredMock( TermLanguageFallbackChain::class, [
			'getFetchLanguageCodes' => [ $userLanguageCode ],
		] );
		$messageLocalizer = new MockMessageLocalizer();
		if ( $config === null ) {
			$config = new HashConfig( [
				'WBQualityConstraintsConstraintCheckedOnMainValueId' => 'Q1',
				'WBQualityConstraintsConstraintCheckedOnQualifiersId' => 'Q2',
				'WBQualityConstraintsConstraintCheckedOnReferencesId' => 'Q3',
				'WBQualityConstraintsAsMainValueId' => 'Q4',
				'WBQualityConstraintsAsQualifiersId' => 'Q5',
				'WBQualityConstraintsAsReferencesId' => 'Q6',
			] );
		}
		return new ViolationMessageRenderer(
			$entityIdFormatter,
			$dataValueFormatter,
			$languageNameUtils,
			$userLanguageCode,
			$languageFallbackChain,
			$messageLocalizer,
			$config,
			$maxListLength
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 */
	public function testRender_simpleMessage() {
		$messageKey = 'wbqc-violation-message-single-value';
		$message = new ViolationMessage( $messageKey );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$this->assertSame( '(' . $messageKey . ')', $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_entityId() {
		$messageKey = 'wbqc-violation-message-no-qualifiers';
		$entityId = new NumericPropertyId( 'P1' );
		$message = ( new ViolationMessage( $messageKey ) )
			->withEntityId( $entityId );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-no-qualifiers: P1)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_entityIdList() {
		$messageKey = 'wbqc-violation-message-unique-value';
		$entityIdList = [ new ItemId( 'Q1' ), new NumericPropertyId( 'P2' ) ];
		$message = ( new ViolationMessage( $messageKey ) )
			->withEntityIdList( $entityIdList );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-unique-value: 2, ' .
			'<ul><li>Q1</li><li>P2</li></ul>, ' .
			'Q1, P2)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_itemIdSnakValue() {
		$messageKey = 'wbqc-violation-message-conflicts-with-claim';
		$itemIdSnakValue = ItemIdSnakValue::someValue();
		$message = ( new ViolationMessage( $messageKey ) )
			->withEntityId( new NumericPropertyId( 'P1' ) )
			->withEntityId( new NumericPropertyId( 'P2' ) )
			->withItemIdSnakValue( $itemIdSnakValue );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-conflicts-with-claim: P1, P2, ' .
			'<span class="wikibase-snakview-variation-somevaluesnak">' .
			'(wikibase-snakview-snaktypeselector-somevalue)</span>)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_itemIdSnakValueList() {
		$messageKey = 'wbqc-violation-message-one-of';
		$valueList = [ ItemIdSnakValue::fromItemId( new ItemId( 'Q1' ) ) ];
		$message = ( new ViolationMessage( $messageKey ) )
			->withEntityId( new NumericPropertyId( 'P1' ) )
			->withItemIdSnakValueList( $valueList );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-one-of: P1, 1, <ul><li>Q1</li></ul>, Q1)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_dataValue() {
		$messageKey = 'wbqc-violation-message-range-quantity-rightopen';
		$dataValue = new StringValue( 'a string' );
		$message = ( new ViolationMessage( $messageKey ) )
			->withEntityId( new NumericPropertyId( 'P1' ) )
			->withDataValue( $dataValue )
			->withDataValue( $dataValue );
		$renderer = $this->newViolationMessageRenderer( null, new StringFormatter() );

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-range-quantity-rightopen: P1, a string, a string)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_dataValueType() {
		$messageKey = 'wbqc-violation-message-value-needed-of-type';
		$dataValueType = 'string';
		$message = ( new ViolationMessage( $messageKey ) )
			->withEntityId( new ItemId( 'Q1' ) )
			->withDataValueType( $dataValueType );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-value-needed-of-type: Q1, (datatypes-type-string))';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_inlineCode() {
		$messageKey = 'wbqc-violation-message-format';
		$code = 'https?://[^/]+/.*';
		$message = ( new ViolationMessage( $messageKey ) )
			->withEntityId( new ItemId( 'Q1' ) )
			->withDataValue( new StringValue( 'ftp://mirror.example/' ) )
			->withInlineCode( $code );
		$renderer = $this->newViolationMessageRenderer( null, new StringFormatter() );

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-format: Q1, ' .
			'ftp://mirror.example/, <code>https?://[^/]+/.*</code>)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_constraintScope() {
		$messageKey = 'wbqc-violation-message-invalid-scope';
		$scope = Context::TYPE_STATEMENT;
		$message = ( new ViolationMessage( $messageKey ) )
			->withConstraintScope( $scope );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-invalid-scope: Q1)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_constraintScopeList() {
		$messageKey = 'wbqc-violation-message-invalid-scope';
		$scopeList = [ Context::TYPE_STATEMENT, Context::TYPE_REFERENCE ];
		$message = ( new ViolationMessage( $messageKey ) )
			->withConstraintScope( Context::TYPE_QUALIFIER )
			->withEntityId( new ItemId( 'Q10' ) )
			->withConstraintScopeList( $scopeList );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-invalid-scope: Q2, Q10, 2, ' .
			'<ul><li>Q1</li><li>Q3</li></ul>, Q1, Q3)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_propertyScope() {
		$messageKey = 'wbqc-violation-message-invalid-scope';
		$scope = Context::TYPE_STATEMENT;
		$message = ( new ViolationMessage( $messageKey ) )
			->withPropertyScope( $scope );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-invalid-scope: Q4)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_propertyScopeList() {
		$messageKey = 'wbqc-violation-message-invalid-scope';
		$scopeList = [ Context::TYPE_STATEMENT, Context::TYPE_REFERENCE ];
		$message = ( new ViolationMessage( $messageKey ) )
			->withPropertyScope( Context::TYPE_QUALIFIER )
			->withEntityId( new ItemId( 'Q10' ) )
			->withPropertyScopeList( $scopeList );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-invalid-scope: Q5, Q10, 2, ' .
			'<ul><li>Q4</li><li>Q6</li></ul>, Q4, Q6)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_language() {
		$messageKey = 'wbqc-violation-message-parameter-single-per-language';
		$languageCode = 'pt';
		$message = ( new ViolationMessage( $messageKey ) )
			->withEntityId( new NumericPropertyId( 'P1' ) )
			->withLanguage( $languageCode );
		$renderer = $this->newViolationMessageRenderer();

		$rendered = $renderer->render( $message );

		$expected = '(wbqc-violation-message-parameter-single-per-language: ' .
			'P1, Portuguese, pt)';
		$this->assertSame( $expected, $rendered );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::render
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderArgument
	 */
	public function testRender_unknownArgumentType() {
		$message = $this->createMock( ViolationMessage::class );
		$message->method( 'getArguments' )
			->willReturn( [ [ 'type' => 'unknown', 'value' => null, 'role' => null ] ] );
		$renderer = $this->newViolationMessageRenderer();

		$this->expectException( InvalidArgumentException::class );
		$renderer->render( $message );
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderList
	 */
	public function testRenderList() {
		$valueList = [ '<any value>', new StringValue( 'any kind of value' ) ];
		$role = null;
		$renderer = $this->newViolationMessageRenderer();
		$renderMock = $this->getMockBuilder( \stdClass::class )
			->addMethods( [ 'render' ] )
			->getMock();
		$renderMock->expects( $this->exactly( 2 ) )
			->method( 'render' )
			->withConsecutive( [ $valueList[0], $role ], [ $valueList[1], $role ] )
			->willReturnCallback( function ( $value, $role ) {
				if ( $value instanceof StringValue ) {
					return [ Message::rawParam( $value->getValue() ) ];
				} else {
					return [ Message::rawParam( $value ) ];
				}
			} );
		$renderFunction = [ $renderMock, 'render' ];

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderList( $valueList, $role, $renderFunction );

		$this->assertSame(
			[
				Message::numParam( 2 ),
				Message::rawParam( '<ul><li><any value></li><li>any kind of value</li></ul>' ),
				Message::rawParam( '<any value>' ),
				Message::rawParam( 'any kind of value' ),
			],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderList
	 */
	public function testRenderList_empty() {
		$valueList = [];
		$role = null;
		$renderer = $this->newViolationMessageRenderer();
		$renderMock = $this->getMockBuilder( \stdClass::class )
			->addMethods( [ 'render' ] )
			->getMock();
		$renderMock->expects( $this->never() )
			->method( 'render' );
		$renderFunction = [ $renderMock, 'render' ];

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderList( $valueList, $role, $renderFunction );

		$this->assertSame(
			[
				Message::numParam( 0 ),
				Message::rawParam( '<ul></ul>' ),
			],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderList
	 */
	public function testRenderList_tooLong() {
		$valueList = [ 'Q1', 'P2', 'Q3' ];
		$role = null;
		$renderer = $this->newViolationMessageRenderer( null, null, null, 2 );
		$renderMock = $this->getMockBuilder( \stdClass::class )
			->addMethods( [ 'render' ] )
			->getMock();
		$renderMock->expects( $this->exactly( 2 ) )
			->method( 'render' )
			->withConsecutive( [ $valueList[0], $role ], [ $valueList[1], $role ] )
			->willReturnCallback( function ( $value, $role ) {
				return [ Message::rawParam( $value ) ];
			} );
		$renderFunction = [ $renderMock, 'render' ];

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderList( $valueList, $role, $renderFunction );

		$this->assertSame(
			[
				Message::numParam( 2 ),
				Message::rawParam( '<ul><li>Q1</li><li>P2</li><li>(ellipsis)</li></ul>' ),
				Message::rawParam( 'Q1' ),
				Message::rawParam( 'P2' ),
			],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderList
	 */
	public function testRenderList_withRole() {
		$valueList = [ '<test item>' ];
		$role = Role::OBJECT;
		$renderer = $this->newViolationMessageRenderer();
		$renderMock = $this->getMockBuilder( \stdClass::class )
			->addMethods( [ 'render' ] )
			->getMock();
		$renderMock->expects( $this->once() )
			->method( 'render' )
			->with( $valueList[0], $role )
			->willReturn( [ Message::rawParam(
				'<span class="wbqc-role wbqc-role-object">' . $valueList[0] . '</span>'
			) ] );
		$renderFunction = [ $renderMock, 'render' ];

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderList( $valueList, $role, $renderFunction );

		$this->assertSame(
			[
				Message::numParam( 1 ),
				Message::rawParam( '<ul><li><span class="wbqc-role wbqc-role-object"><test item></span></li></ul>' ),
				Message::rawParam( '<span class="wbqc-role wbqc-role-object"><test item></span>' ),
			],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderEntityId
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderEntityId() {
		$entityId = new ItemId( 'Q1' );
		$role = null;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter->expects( $this->once() )
			->method( 'formatEntityId' )
			->with( $entityId )
			->willReturn( '<test property>' );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderEntityId( $entityId, $role );

		$this->assertSame(
			[ Message::rawParam( '<test property>' ) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderEntityId
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderEntityId_withRole() {
		$entityId = new NumericPropertyId( 'P1' );
		$role = Role::PREDICATE;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter
			->method( 'formatEntityId' )
			->willReturn( '<test property>' );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderEntityId( $entityId, $role );

		$this->assertSame(
			[ Message::rawParam( '<span class="wbqc-role wbqc-role-predicate"><test property></span>' ) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderEntityIdList
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderEntityIdList() {
		$entityIdList = [ new ItemId( 'Q1' ), new NumericPropertyId( 'P2' ) ];
		$role = null;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter->expects( $this->exactly( 2 ) )
			->method( 'formatEntityId' )
			->willReturnCallback( [ new PlainEntityIdFormatter(), 'formatEntityId' ] );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderEntityIdList( $entityIdList, $role );

		$this->assertSame(
			[
				Message::numParam( 2 ),
				Message::rawParam( '<ul><li>Q1</li><li>P2</li></ul>' ),
				Message::rawParam( 'Q1' ),
				Message::rawParam( 'P2' ),
			],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderEntityIdList
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderEntityIdList_withRole() {
		$entityIdList = [ new ItemId( 'Q1' ) ];
		$role = Role::OBJECT;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter
			->method( 'formatEntityId' )
			->willReturn( '<test item>' );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderEntityIdList( $entityIdList, $role );

		$this->assertSame(
			[
				Message::numParam( 1 ),
				Message::rawParam( '<ul><li><span class="wbqc-role wbqc-role-object"><test item></span></li></ul>' ),
				Message::rawParam( '<span class="wbqc-role wbqc-role-object"><test item></span>' ),
			],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderItemIdSnakValue
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderItemIdSnakValue_itemId() {
		$itemId = new ItemId( 'Q1' );
		$itemIdSnakValue = ItemIdSnakValue::fromItemId( $itemId );
		$role = null;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter->expects( $this->once() )
			->method( 'formatEntityId' )
			->with( $itemId )
			->willReturn( '<test item>' );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderItemIdSnakValue( $itemIdSnakValue, $role );

		$this->assertSame(
			[ Message::rawParam( '<test item>' ) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderItemIdSnakValue
	 */
	public function testRenderItemIdSnakValue_someValue() {
		$itemIdSnakValue = ItemIdSnakValue::someValue();
		$role = null;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter
			->expects( $this->never() )
			->method( 'formatEntityId' );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderItemIdSnakValue( $itemIdSnakValue, $role );

		$this->assertSame(
			[ Message::rawParam(
				'<span class="wikibase-snakview-variation-somevaluesnak">' .
					'(wikibase-snakview-snaktypeselector-somevalue)' .
					'</span>'
			) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderItemIdSnakValue
	 */
	public function testRenderItemIdSnakValue_noValue() {
		$itemIdSnakValue = ItemIdSnakValue::noValue();
		$role = null;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter
			->expects( $this->never() )
			->method( 'formatEntityId' );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderItemIdSnakValue( $itemIdSnakValue, $role );

		$this->assertSame(
			[ Message::rawParam(
				'<span class="wikibase-snakview-variation-novaluesnak">' .
				'(wikibase-snakview-snaktypeselector-novalue)' .
				'</span>'
			) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderItemIdSnakValue
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderItemIdSnakValue_withRole() {
		$itemId = new ItemId( 'Q1' );
		$itemIdSnakValue = ItemIdSnakValue::fromItemId( $itemId );
		$role = Role::OBJECT;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter
			->method( 'formatEntityId' )
			->willReturn( '<test item>' );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderItemIdSnakValue( $itemIdSnakValue, $role );

		$this->assertSame(
			[ Message::rawParam( '<span class="wbqc-role wbqc-role-object"><test item></span>' ) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderItemIdSnakValueList
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderItemIdSnakValueList() {
		$valueList = [
			ItemIdSnakValue::fromItemId( new ItemId( 'Q1' ) ),
			ItemIdSnakValue::fromItemId( new ItemId( 'Q2' ) ),
		];
		$role = null;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter->expects( $this->exactly( 2 ) )
			->method( 'formatEntityId' )
			->willReturnCallback( [ new PlainEntityIdFormatter(), 'formatEntityId' ] );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderItemIdSnakValueList( $valueList, $role );

		$this->assertSame(
			[
				Message::numParam( 2 ),
				Message::rawParam( '<ul><li>Q1</li><li>Q2</li></ul>' ),
				Message::rawParam( 'Q1' ),
				Message::rawParam( 'Q2' ),
			],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderItemIdSnakValueList
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderItemIdSnakValueList_withRole() {
		$valueList = [ ItemIdSnakValue::fromItemId( new ItemId( 'Q1' ) ) ];
		$role = Role::OBJECT;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter
			->method( 'formatEntityId' )
			->willReturn( '<test item>' );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderItemIdSnakValueList( $valueList, $role );

		$this->assertSame(
			[
				Message::numParam( 1 ),
				Message::rawParam( '<ul><li><span class="wbqc-role wbqc-role-object"><test item></span></li></ul>' ),
				Message::rawParam( '<span class="wbqc-role wbqc-role-object"><test item></span>' ),
			],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderDataValue
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderDataValue() {
		$dataValue = new StringValue( 'a&nbsp;string' );
		$role = null;
		$dataValueFormatter = $this->createMock( ValueFormatter::class );
		$dataValueFormatter->expects( $this->once() )
			->method( 'format' )
			->with( $dataValue )
			->willReturn( '<a&amp;nbsp;string>' );
		$renderer = $this->newViolationMessageRenderer( null, $dataValueFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderDataValue( $dataValue, $role );

		$this->assertSame(
			[ Message::rawParam( '<a&amp;nbsp;string>' ) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderDataValue
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderDataValue_withRole() {
		$dataValue = new StringValue( 'test' );
		$role = Role::OBJECT;
		$dataValueFormatter = $this->createMock( ValueFormatter::class );
		$dataValueFormatter
			->method( 'format' )
			->willReturn( 'test' );
		$renderer = $this->newViolationMessageRenderer( null, $dataValueFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderDataValue( $dataValue, $role );

		$this->assertSame(
			[ Message::rawParam( '<span class="wbqc-role wbqc-role-object">test</span>' ) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderDataValueType
	 */
	public function testRenderDataValueType_string() {
		$dataValueType = 'string';
		$role = null;
		$renderer = $this->newViolationMessageRenderer();

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderDataValueType( $dataValueType, $role );

		$this->assertSame(
			[ Message::rawParam( '(datatypes-type-string)' ) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderDataValueType
	 */
	public function testRenderDataValueType_entityId() {
		$dataValueType = 'wikibase-entityid';
		$role = null;
		$renderer = $this->newViolationMessageRenderer();

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderDataValueType( $dataValueType, $role );

		$this->assertSame(
			[ Message::rawParam( '(wbqc-dataValueType-wikibase-entityid)' ) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderInlineCode
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderInlineCode() {
		$code = 'https?://[^/]+/.*';
		$role = null;
		$renderer = $this->newViolationMessageRenderer();

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderInlineCode( $code, $role );

		$this->assertSame(
			[ Message::rawParam( '<code>https?://[^/]+/.*</code>' ) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderInlineCode
	 */
	public function testRenderInlineCode_htmlEscape() {
		$code = '<script>alert("im in ur html")</script>';
		$role = null;
		$renderer = $this->newViolationMessageRenderer();

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderInlineCode( $code, $role );

		$this->assertSame(
			[ Message::rawParam(
				'<code>' .
					'&lt;script&gt;alert(&quot;im in ur html&quot;)&lt;/script&gt;' .
					'</code>'
			) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderInlineCode
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderInlineCode_withRole() {
		$code = 'https?://[^/]+/.*';
		$role = Role::CONSTRAINT_PARAMETER_VALUE;
		$renderer = $this->newViolationMessageRenderer();

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderInlineCode( $code, $role );

		$this->assertSame(
			[ Message::rawParam(
				'<span class="wbqc-role wbqc-role-constraint-parameter-value">' .
					'<code>https?://[^/]+/.*</code>' .
					'</span>'
			) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderConstraintScope
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 * @dataProvider provideConstraintScopes
	 */
	public function testRenderConstraintScope( $contextType, $itemIdSerialization, $returnValue ) {
		$scope = $contextType;
		$role = null;
		$itemId = new ItemId( $itemIdSerialization );
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter->expects( $this->once() )
			->method( 'formatEntityId' )
			->with( $itemId )
			->willReturn( $returnValue );
		$config = new HashConfig( [
			'WBQualityConstraintsConstraintCheckedOnMainValueId' => 'Q10',
			'WBQualityConstraintsConstraintCheckedOnQualifiersId' => 'Q20',
			'WBQualityConstraintsConstraintCheckedOnReferencesId' => 'Q30',
		] );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter, null, $config );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderConstraintScope( $scope, $role );

		$this->assertSame(
			[ Message::rawParam( $returnValue ) ],
			$params
		);
	}

	public static function provideConstraintScopes() {
		return [
			[ Context::TYPE_STATEMENT, 'Q10', 'statement scope' ],
			[ Context::TYPE_QUALIFIER, 'Q20', 'qualifier scope' ],
			[ Context::TYPE_REFERENCE, 'Q30', 'reference scope' ],
		];
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderConstraintScope
	 */
	public function testRenderConstraintScope_unknown() {
		$scope = 'some unknown scope';
		$role = null;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter->expects( $this->never() )
			->method( 'formatEntityId' );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderConstraintScope( $scope, $role );

		$this->assertSame(
			[ Message::rawParam(
				'<span class="wikibase-snakview-variation-somevaluesnak">' .
					'(wikibase-snakview-snaktypeselector-somevalue)' .
					'</span>'
			) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderConstraintScope
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderConstraintScope_withRole() {
		$scope = Context::TYPE_STATEMENT;
		$role = Role::CONSTRAINT_PARAMETER_VALUE;
		$renderer = $this->newViolationMessageRenderer();

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderConstraintScope( $scope, $role );

		$this->assertSame(
			[ Message::rawParam(
				'<span class="wbqc-role wbqc-role-constraint-parameter-value">Q1</span>'
			) ],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderConstraintScopeList
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderConstraintScopeList() {
		$scopeList = [ Context::TYPE_STATEMENT, Context::TYPE_REFERENCE ];
		$role = null;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter->expects( $this->exactly( 2 ) )
			->method( 'formatEntityId' )
			->willReturnCallback( [ new PlainEntityIdFormatter(), 'formatEntityId' ] );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderConstraintScopeList( $scopeList, $role );

		$this->assertSame(
			[
				Message::numParam( 2 ),
				Message::rawParam( '<ul><li>Q1</li><li>Q3</li></ul>' ),
				Message::rawParam( 'Q1' ),
				Message::rawParam( 'Q3' ),
			],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderConstraintScopeList
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::addRole
	 */
	public function testRenderConstraintScopeList_withRole() {
		$scopeList = [ Context::TYPE_STATEMENT ];
		$role = Role::CONSTRAINT_PARAMETER_VALUE;
		$entityIdFormatter = $this->createMock( EntityIdFormatter::class );
		$entityIdFormatter
			->method( 'formatEntityId' )
			->willReturn( '<test scope>' );
		$renderer = $this->newViolationMessageRenderer( $entityIdFormatter );

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderConstraintScopeList( $scopeList, $role );

		$html = '<span class="wbqc-role wbqc-role-constraint-parameter-value"><test scope></span>';
		$this->assertSame(
			[
				Message::numParam( 1 ),
				Message::rawParam( '<ul><li>' . $html . '</li></ul>' ),
				Message::rawParam( $html ),
			],
			$params
		);
	}

	/**
	 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer::renderLanguage
	 */
	public function testRenderLanguage() {
		$languageCode = 'pt';
		$role = null;
		$renderer = $this->newViolationMessageRenderer();

		$params = TestingAccessWrapper::newFromObject( $renderer )
			->renderLanguage( $languageCode, $role );

		$this->assertSame(
			[
				Message::rawParam( 'Portuguese' ),
				Message::plaintextParam( 'pt' ),
			],
			$params
		);
	}

}
