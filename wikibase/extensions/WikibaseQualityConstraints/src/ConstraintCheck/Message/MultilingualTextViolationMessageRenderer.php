<?php

declare( strict_types = 1 );

namespace WikibaseQuality\ConstraintReport\ConstraintCheck\Message;

use DataValues\MultilingualTextValue;
use Message;

/**
 * Render a {@link ViolationMessage},
 * which may have a multilingual text as the last argument,
 * into a localized string.
 * (For all other messages, this falls back to the base {@link ViolationMessageRenderer}.)
 *
 * Note that this is only supported for specific message keys,
 * where an alternative message key is known
 * which is used if the text is not available in the user’s language.
 * Currently, the only such message key is 'wbqc-violation-message-format-clarification'
 * (falling back to 'wbqc-violation-message-format').
 *
 * @license GPL-2.0-or-later
 */
class MultilingualTextViolationMessageRenderer extends ViolationMessageRenderer {

	private const ALTERNATIVE_MESSAGE_KEYS = [
		'wbqc-violation-message-format-clarification' => 'wbqc-violation-message-format',
	];

	public function render( ViolationMessage $violationMessage ): string {
		if ( !array_key_exists( $violationMessage->getMessageKey(), self::ALTERNATIVE_MESSAGE_KEYS ) ) {
			return parent::render( $violationMessage );
		}

		$arguments = $violationMessage->getArguments();
		$multilingualTextArgument = array_pop( $arguments );
		$multilingualTextParams = $this->renderMultilingualText(
			// @phan-suppress-next-line PhanTypeArraySuspiciousNullable TODO Ensure this is not an actual issue
			$multilingualTextArgument['value'],
			// @phan-suppress-next-line PhanTypeArraySuspiciousNullable
			$multilingualTextArgument['role']
		);

		$paramsLists = [ [] ];
		foreach ( $arguments as $argument ) {
			$paramsLists[] = $this->renderArgument( $argument );
		}
		$regularParams = call_user_func_array( 'array_merge', $paramsLists );

		if ( $multilingualTextParams === null ) {
			return $this->messageLocalizer
				->msg( self::ALTERNATIVE_MESSAGE_KEYS[$violationMessage->getMessageKey()] )
				->params( $regularParams )
				->escaped();
		} else {
			return $this->messageLocalizer
				->msg( $violationMessage->getMessageKey() )
				->params( $regularParams )
				->params( $multilingualTextParams )
				->escaped();
		}
	}

	/**
	 * @param MultilingualTextValue $text
	 * @param string|null $role one of the Role::* constants
	 * @return array[]|null list of parameters as accepted by Message::params(),
	 * or null if the text is not available in the user’s language
	 */
	protected function renderMultilingualText( MultilingualTextValue $text, ?string $role ): ?array {
		$texts = $text->getTexts();
		foreach ( $this->languageFallbackChain->getFetchLanguageCodes() as $languageCode ) {
			if ( array_key_exists( $languageCode, $texts ) ) {
				return [ Message::rawParam( $this->addRole(
					htmlspecialchars( $texts[$languageCode]->getText() ),
					$role
				) ) ];
			}
		}

		return null;
	}

}
