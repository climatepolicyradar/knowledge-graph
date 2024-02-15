<?php

namespace WikibaseQuality\ConstraintReport\Tests\Checker\FormatChecker;

use Config;
use DataValues\StringValue;
use HashConfig;
use MediaWiki\MediaWikiServices;
use MediaWiki\Shell\ShellboxClientFactory;
use MultiConfig;
use Shellbox\Client;
use Shellbox\ShellboxError;
use Wikibase\DataModel\Entity\EntityIdValue;
use Wikibase\DataModel\Entity\ItemId;
use Wikibase\DataModel\Entity\NumericPropertyId;
use Wikibase\DataModel\Snak\PropertyNoValueSnak;
use Wikibase\DataModel\Snak\PropertyValueSnak;
use Wikibase\DataModel\Tests\NewItem;
use Wikibase\DataModel\Tests\NewStatement;
use WikibaseQuality\ConstraintReport\Constraint;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\FormatChecker;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Context\MainSnakContext;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\ConstraintParameterException;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\DummySparqlHelper;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Helper\SparqlHelper;
use WikibaseQuality\ConstraintReport\Tests\ConstraintParameters;
use WikibaseQuality\ConstraintReport\Tests\Fake\FakeSnakContext;
use WikibaseQuality\ConstraintReport\Tests\ResultAssertions;

/**
 * @covers WikibaseQuality\ConstraintReport\ConstraintCheck\Checker\FormatChecker
 *
 * @group WikibaseQualityConstraints
 *
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class FormatCheckerTest extends \MediaWikiIntegrationTestCase {

	use ConstraintParameters;
	use ResultAssertions;

	public static function provideFormatConstraintCompliance() {
		$imdbRegex = '(tt|nm|ch|co|ev)\d{7}';
		$taxonRegex = '(|somevalue|novalue|.*virus.*|.*viroid.*|.*phage.*|((×)?[A-Z]([a-z]+-)?[a-z]+('
			. '( [A-Z]?[a-z]+)|'
			. '( ([a-z]+-)?([a-z]+-)?[a-z]+)|'
			. '( ×([a-z]+-)?([a-z]+-)?([a-z]+-)?([a-z]+-)?[a-z]+)|'
			. '( \([A-Z][a-z]+\) [a-z]+)|'
			. "( (‘|')[A-Z][a-z]+(('|’)s)?( de)?( [A-Z][a-z]+(-([A-Z])?[a-z]+)*)*('|’)*)|"
			. '( ×| Group| (sub)?sp\.| (con)?(sub)?(notho)?var\.| (sub)?ser\.| (sub)?sect\.|'
			. ' subg\.| (sub)?f\.))*))';
		return [
			[ $imdbRegex, 'nm0001398' ],
			[ $imdbRegex, 'tt1234567' ],
			[ $imdbRegex, 'ch7654321' ],
			[ $imdbRegex, 'ev7777777' ],
			[ $taxonRegex, 'Populus × canescens' ],
			[ $taxonRegex, 'Encephalartos friderici-guilielmi' ],
			[ $taxonRegex, 'Eruca vesicaria subsp. sativa' ],
			[ $taxonRegex, 'Euxoa (Chorizagrotis) lidia' ],
			[ '[Α-Ω]{3,5}', 'ΚΑΤΛ' ],
		];
	}

	/** @dataProvider provideFormatConstraintCompliance
	 * @param string $pattern
	 * @param string $text
	 */
	public function testFormatConstraintComplianceSparql( string $pattern, string $text ) {
		$config = $this->getMultiConfig( [ 'WBQualityConstraintsFormatCheckerShellboxRatio' => 0 ] );
		$value = new StringValue( $text );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P345' ), $value );
		$formatChecker = $this->getChecker( $config );
		$result = $formatChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->formatParameter( $pattern ) )
		);
		$this->assertCompliance( $result );
	}

	/** @dataProvider provideFormatConstraintCompliance
	 * @param string $pattern
	 * @param string $text
	 */
	public function testFormatConstraintComplianceShellbox( string $pattern, string $text ) {
		$config = $this->getMultiConfig( [ 'WBQualityConstraintsFormatCheckerShellboxRatio' => 1 ] );
		$value = new StringValue( $text );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P345' ), $value );
		$formatChecker = $this->getChecker( $config );
		$result = $formatChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->formatParameter( $pattern ) )
		);
		$this->assertCompliance( $result );
	}

	public static function provideFormatConstraintViolation() {
		$imdbRegex = '(tt|nm|ch|co|ev)\d{7}';
		$taxonRegex = '(|somevalue|novalue|.*virus.*|.*viroid.*|.*phage.*|((×)?[A-Z]([a-z]+-)?[a-z]+('
			. '( [A-Z]?[a-z]+)|'
			. '( ([a-z]+-)?([a-z]+-)?[a-z]+)|'
			. '( ×([a-z]+-)?([a-z]+-)?([a-z]+-)?([a-z]+-)?[a-z]+)|'
			. '( \([A-Z][a-z]+\) [a-z]+)|'
			. "( (‘|')[A-Z][a-z]+(('|’)s)?( de)?( [A-Z][a-z]+(-([A-Z])?[a-z]+)*)*('|’)*)|"
			. '( ×| Group| (sub)?sp\.| (con)?(sub)?(notho)?var\.| (sub)?ser\.| (sub)?sect\.|'
			. ' subg\.| (sub)?f\.))*))';
		return [
			[ $imdbRegex, 'nm88888888' ],
			[ $imdbRegex, 'nmabcdefg' ],
			[ $imdbRegex, 'ab0001398' ],
			[ $imdbRegex, '123456789' ],
			[ $imdbRegex, 'nm000139' ],
			[ $imdbRegex, 'nmnm0001398' ],
			[ $taxonRegex, 'Hepatitis A' ],
			[ $taxonRegex, 'Symphysodon (Cichlidae)' ],
			[ $taxonRegex, 'eukaryota' ],
			[ $taxonRegex, 'Plantago maritima agg.' ],
			[ $taxonRegex, 'Deinococcus-Thermus' ],
			[ $taxonRegex, 'Escherichia coli O157:H7' ],
			[ 'ab|cd', 'abcd' ],
		];
	}

	/** @dataProvider provideFormatConstraintViolation
	 * @param string $pattern
	 * @param string $text
	 */
	public function testFormatConstraintViolationSparql( string $pattern, string $text ) {
		$config = $this->getMultiConfig( [ 'WBQualityConstraintsFormatCheckerShellboxRatio' => 0 ] );
		$value = new StringValue( $text );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P345' ), $value );
		$formatChecker = $this->getChecker();
		$result = $formatChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->formatParameter( $pattern ) )
		);
		$this->assertViolation( $result, 'wbqc-violation-message-format-clarification' );
	}

	/** @dataProvider provideFormatConstraintViolation
	 * @param string $pattern
	 * @param string $text
	 */
	public function testFormatConstraintViolationShellbox( string $pattern, string $text ) {
		$config = $this->getMultiConfig( [ 'WBQualityConstraintsFormatCheckerShellboxRatio' => 1 ] );
		$value = new StringValue( $text );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P345' ), $value );
		$formatChecker = $this->getChecker( $config );
		$result = $formatChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->formatParameter( $pattern ) )
		);
		$this->assertViolation( $result, 'wbqc-violation-message-format-clarification' );
	}

	public function testFormatConstraintWithSyntaxClarification() {
		$syntaxClarificationId = self::getDefaultConfig()
			->get( 'WBQualityConstraintsSyntaxClarificationId' );
		$statement = NewStatement::forProperty( $syntaxClarificationId )
			->withValue( '' )
			->build();

		$formatChecker = $this->getChecker();
		$result = $formatChecker->checkConstraint(
			new FakeSnakContext( $statement->getMainSnak() ),
			$this->getConstraintMock( array_merge(
				$this->formatParameter( '.+' ),
				$this->syntaxClarificationParameter( 'en', 'nonempty' )
			) )
		);

		$this->assertViolation( $result, 'wbqc-violation-message-format-clarification' );
	}

	public function testFormatConstraintNoStringValue() {
		$pattern = '(tt|nm|ch|co|ev)\d{7}';

		$value = new EntityIdValue( new ItemId( 'Q1' ) );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P345' ), $value );

		$formatChecker = $this->getChecker();
		$result = $formatChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->formatParameter( $pattern ) )
		);
		$this->assertViolation( $result );
	}

	public function testFormatConstraintNoValueSnak() {
		$pattern = ".";
		$snak = new PropertyNoValueSnak( new NumericPropertyId( 'P1' ) );

		$formatChecker = $this->getChecker();
		$result = $formatChecker->checkConstraint(
			new FakeSnakContext( $snak ),
			$this->getConstraintMock( $this->formatParameter( $pattern ) )
		);
		$this->assertCompliance( $result );
	}

	public function testFormatConstraintWithoutSparql() {
		$config = $this->getMultiConfig( [ 'WBQualityConstraintsFormatCheckerShellboxRatio' => 0 ] );
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), new StringValue( '' ) );
		$constraint = $this->getConstraintMock( $this->formatParameter( '.' ) );
		$checker = new FormatChecker(
			$this->getConstraintParameterParser(),
			$config,
			new DummySparqlHelper(),
			MediaWikiServices::getInstance()->getShellboxClientFactory()
		);

		$result = $checker->checkConstraint(
			new FakeSnakContext( $snak ),
			$constraint
		);

		$this->assertTodoViolation( $result );
	}

	public function testFormatConstraintDisabled() {
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), new StringValue( '' ) );
		$constraint = $this->getConstraintMock( $this->formatParameter( '.' ) );
		$sparqlHelper = $this->getMockBuilder( SparqlHelper::class )
					  ->disableOriginalConstructor()
					  ->onlyMethods( [ 'matchesRegularExpression' ] )
					  ->getMock();
		$sparqlHelper->expects( $this->never() )->method( 'matchesRegularExpression' );
		$checker = new FormatChecker(
			$this->getConstraintParameterParser(),
			new HashConfig( [ 'WBQualityConstraintsCheckFormatConstraint' => false ] ),
			$sparqlHelper,
			MediaWikiServices::getInstance()->getShellboxClientFactory()
		);

		$result = $checker->checkConstraint(
			new FakeSnakContext( $snak ),
			$constraint
		);

		$this->assertTodo( $result );
	}

	public function testFormatConstraintShellboxDisabled() {
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), new StringValue( '' ) );
		$constraint = $this->getConstraintMock( $this->formatParameter( '.' ) );
		$sparqlHelper = $this->createMock( SparqlHelper::class );
		$shellboxClientFactory = $this->getMockBuilder( ShellboxClientFactory::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'isEnabled' ] )
			->getMock();
		$shellboxClientFactory->method( 'isEnabled' )
			->willReturn( false );
		$checker = new FormatChecker(
			$this->getConstraintParameterParser(),
			new HashConfig( [
				'WBQualityConstraintsFormatCheckerShellboxRatio' => 1,
				'WBQualityConstraintsCheckFormatConstraint' => true,
			] ),
			$sparqlHelper,
			$shellboxClientFactory
		);

		$result = $checker->checkConstraint(
			new FakeSnakContext( $snak ),
			$constraint
		);

		$this->assertTodo( $result );
	}

	public function testFormatConstraintShellboxError() {
		$snak = new PropertyValueSnak( new NumericPropertyId( 'P1' ), new StringValue( '' ) );
		$sparqlHelper = $this->createMock( SparqlHelper::class );
		$shellboxClient = $this->getMockBuilder( Client::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'call' ] )
			->getMock();
		$shellboxClient->method( 'call' )
			->willThrowException( new ShellboxError() );
		$shellboxClientFactory = $this->getMockBuilder( ShellboxClientFactory::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'getClient', 'isEnabled' ] )
			->getMock();
		$shellboxClientFactory->method( 'isEnabled' )
			->willReturn( true );
		$shellboxClientFactory->method( 'getClient' )
			->willReturn( $shellboxClient );
		$constraint = $this->getConstraintMock( $this->formatParameter( '.' ) );
		$checker = new FormatChecker(
			$this->getConstraintParameterParser(),
			new HashConfig( [
				'WBQualityConstraintsFormatCheckerShellboxRatio' => 1,
				'WBQualityConstraintsSparqlMaxMillis' => 100,
				'WBQualityConstraintsCheckFormatConstraint' => true,
			] ),
			$sparqlHelper,
			$shellboxClientFactory
		);

		$this->expectException( ConstraintParameterException::class );

		$checker->checkConstraint(
			new FakeSnakContext( $snak ),
			$constraint
		);
	}

	public function testFormatConstraintDeprecatedStatement() {
		$statement = NewStatement::forProperty( 'P1' )
				   ->withValue( 'abc' )
				   ->withDeprecatedRank()
				   ->build();
		$constraint = $this->getConstraintMock( $this->formatParameter( 'a.b.' ) );
		$entity = NewItem::withId( 'Q1' )
				->build();

		$formatChecker = $this->getChecker();
		$checkResult = $formatChecker->checkConstraint(
			new MainSnakContext( $entity, $statement ),
			$constraint
		);

		// this constraint is still checked on deprecated statements
		$this->assertViolation( $checkResult, 'wbqc-violation-message-format-clarification' );
	}

	public function testCheckConstraintParameters() {
		$constraint = $this->getConstraintMock( [] );

		$formatChecker = $this->getChecker();
		$result = $formatChecker->checkConstraintParameters( $constraint );

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
			 ->willReturn( 'Q21502404' );

		return $mock;
	}

	private function getChecker( ?Config $config = null ): FormatChecker {
		$config = $config ?? self::getDefaultConfig();
		$sparqlHelper = $this->getMockBuilder( SparqlHelper::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'matchesRegularExpression' ] )
			->getMock();
		$sparqlHelper->method( 'matchesRegularExpression' )
			->willReturnCallback(
				function ( $text, $regex ) {
					$pattern = '/^(?:' . str_replace( '/', '\/', $regex ) . ')$/u';
					return preg_match( $pattern, $text );
				}
			);
		$shellboxClient = $this->getMockBuilder( Client::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'call' ] )
			->getMock();
		$shellboxClient->method( 'call' )
			->willReturnCallback(
				function ( $route, $func_name, $args ) {
					return call_user_func_array( $func_name, $args );
				}
			);
		$shellboxClientFactory = $this->getMockBuilder( ShellboxClientFactory::class )
			->disableOriginalConstructor()
			->onlyMethods( [ 'getClient', 'isEnabled' ] )
			->getMock();
		$shellboxClientFactory->method( 'isEnabled' )
			->willReturn( true );
		$shellboxClientFactory->method( 'getClient' )
			->willReturn( $shellboxClient );
		return new FormatChecker(
			$this->getConstraintParameterParser(),
			$config,
			$sparqlHelper,
			$shellboxClientFactory
		);
	}

	private function getMultiConfig( array $overrides = [] ): Config {
		return new MultiConfig( [
			new HashConfig( $overrides ),
			self::getDefaultConfig(),
			MediaWikiServices::getInstance()->getMainConfig(),
		] );
	}

}
