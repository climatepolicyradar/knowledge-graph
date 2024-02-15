<?php

namespace WikibaseQuality\ConstraintReport\Html;

use Html;
use HtmlArmor;
use InvalidArgumentException;
use Wikimedia\Assert\Assert;

/**
 * @author BP2014N1
 * @license GPL-2.0-or-later
 */
class HtmlTableHeaderBuilder {

	/**
	 * Html content of the header
	 *
	 * @var string|HtmlArmor
	 */
	private $content;

	/**
	 * Determines, whether the column should be sortable or not.
	 *
	 * @var bool
	 */
	private $isSortable;

	/**
	 * @param string|HtmlArmor $content
	 * @param bool $isSortable
	 *
	 * @throws InvalidArgumentException
	 */
	public function __construct( $content, $isSortable = false ) {
		Assert::parameterType( [ 'string', HtmlArmor::class ], $content, '$content' );
		Assert::parameterType( 'boolean', $isSortable, '$isSortable' );

		$this->content = $content;
		$this->isSortable = $isSortable;
	}

	/**
	 * @return string HTML
	 */
	public function getContent() {
		return HtmlArmor::getHtml( $this->content );
	}

	/**
	 * @return bool
	 */
	public function getIsSortable() {
		return $this->isSortable;
	}

	/**
	 * Returns header as html.
	 *
	 * @return string HTML
	 */
	public function toHtml() {
		$attributes = [ 'role' => 'columnheader button' ];

		if ( !$this->isSortable ) {
			$attributes['class'] = 'unsortable';
		}

		return Html::rawElement( 'th', $attributes, $this->getContent() );
	}

}
