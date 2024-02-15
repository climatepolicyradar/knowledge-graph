<?php

namespace WikibaseQuality\ConstraintReport\Tests;

use HashConfig;
use MediaWiki\MediaWikiServices;
use Message;
use MockMessageLocalizer;
use Wikibase\DataModel\Services\EntityId\PlainEntityIdFormatter;
use Wikibase\Lib\Formatters\UnDeserializableValueFormatter;
use Wikibase\Lib\TermLanguageFallbackChain;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Message\ViolationMessageRenderer;
use WikibaseQuality\ConstraintReport\ConstraintCheck\Result\CheckResult;

/**
 * Assertions on the status of a {@link CheckResult}.
 *
 * {@link assertTodoCompliance} and {@link assertTodoViolation}
 * are functionally identical to {@link assertTodo},
 * but allow the test author to record the intended outcome even before it is implemented;
 * once the feature is implemented, one only needs to replace ‘assertTodo’ with ‘assert’
 * in order to use the existing test cases.
 *
 * @author Lucas Werkmeister
 * @license GPL-2.0-or-later
 */
trait ResultAssertions {

	private function renderMessage( CheckResult $result ) {
		$resultMessage = $result->getMessage();
		if ( $resultMessage === null ) {
			return '';
		}

		$userLanguageCode = 'qqx';
		$languageFallbackChain = $this->createMock( TermLanguageFallbackChain::class );
		$languageFallbackChain->method( 'getFetchLanguageCodes' )
			->willReturn( [ $userLanguageCode ] );

		$renderer = new ViolationMessageRenderer(
			new PlainEntityIdFormatter(),
			new UnDeserializableValueFormatter(),
			MediaWikiServices::getInstance()->getLanguageNameUtils(),
			$userLanguageCode,
			$languageFallbackChain,
			new MockMessageLocalizer(),
			new HashConfig( [
				'WBQualityConstraintsConstraintCheckedOnMainValueId' => 'Q1',
				'WBQualityConstraintsConstraintCheckedOnQualifiersId' => 'Q2',
				'WBQualityConstraintsConstraintCheckedOnReferencesId' => 'Q3',
				'WBQualityConstraintsAsMainValueId' => 'Q4',
				'WBQualityConstraintsAsQualifiersId' => 'Q5',
				'WBQualityConstraintsAsReferencesId' => 'Q6',
			] )
		);
		return $renderer->render(
			$resultMessage
		);
	}

	/**
	 * Assert that $result indicates compliance with a constraint.
	 *
	 * @param CheckResult $result
	 */
	public function assertCompliance( CheckResult $result ) {
		$this->assertSame(
			CheckResult::STATUS_COMPLIANCE,
			$result->getStatus(),
			'Check should comply; message: ' . $this->renderMessage( $result )
		);
	}

	/**
	 * Assert that $result indicates violation of a constraint.
	 *
	 * @param CheckResult $result
	 * @param string|null $messageKey If present, additionally assert that the violation message
	 *                            matches the message given by this key.
	 */
	public function assertViolation( CheckResult $result, $messageKey = null ) {
		$this->assertSame(
			CheckResult::STATUS_VIOLATION,
			$result->getStatus(),
			'Check should not comply'
		);
		$resultMessage = $result->getMessage();
		$resultMessageKey = $resultMessage->getMessageKey();
		$this->assertTrue(
			( new Message( $resultMessageKey ) )->inLanguage( 'en' )->exists(),
			"Message should not refer to a non-existing message key (⧼{$resultMessageKey}⧽)."
		);
		if ( $messageKey !== null ) {
			$this->assertSame(
				$messageKey,
				$resultMessageKey,
				"Violation message should be ⧼{$messageKey}⧽."
			);
		}
	}

	/**
	 * Assert that $result indicates an unimplemented constraint.
	 *
	 * @param CheckResult $result
	 */
	public function assertTodo( CheckResult $result ) {
		$this->assertSame(
			CheckResult::STATUS_TODO,
			$result->getStatus(),
			'Check is not implemented.'
		);
	}

	/**
	 * Assert that $result indicates an unimplemented constraint,
	 * and also record that $result should indicate compliance once the constraint is implemented.
	 *
	 * @param CheckResult $result
	 * @see assertTodo
	 * @see assertCompliance
	 */
	public function assertTodoCompliance( CheckResult $result ) {
		$this->assertTodo( $result );
	}

	/**
	 * Assert that $result indicates an unimplemented constraint,
	 * and also record that $result should indicate violation once the constraint is implemented.
	 *
	 * @param CheckResult $result
	 * @param string|null $messageKey
	 * @see assertTodo
	 * @see assertViolation
	 */
	public function assertTodoViolation( CheckResult $result, $messageKey = null ) {
		$this->assertTodo( $result );
	}

	/**
	 * Assert that $result indicates a skipped constraint check due to deprecated statement rank.
	 *
	 * @param CheckResult $result
	 */
	public function assertDeprecation( CheckResult $result ) {
		$this->assertSame(
			CheckResult::STATUS_DEPRECATED,
			$result->getStatus(),
			'Check should indicate deprecation; message: ' . $this->renderMessage( $result )
		);
	}

	/**
	 * Assert that $result indicates a skipped constraint check
	 * due to the snak not being within the scope of the constraint.
	 *
	 * @param CheckResult $result
	 */
	public function assertNotInScope( CheckResult $result ) {
		$this->assertSame(
			CheckResult::STATUS_NOT_IN_SCOPE,
			$result->getStatus(),
			'Check should indicate that snak is out of scope of constraint; message: ' .
				$this->renderMessage( $result )
		);
	}

	/**
	 * Assert that $result indicates a skipped constraint check
	 * due to invalid constraint parameters.
	 */
	public function assertBadParameters( CheckResult $result ): void {
		$this->assertSame(
			CheckResult::STATUS_BAD_PARAMETERS,
			$result->getStatus(),
			'Check should indicate that snak is out of scope of constraint; message: ' .
				$this->renderMessage( $result )
		);
	}

}
