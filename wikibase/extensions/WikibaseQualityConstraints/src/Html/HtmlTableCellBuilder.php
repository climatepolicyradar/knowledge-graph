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
class HtmlTableCellBuilder {

	/**
	 * Html content of the cell.
	 *
	 * @var string|HtmlArmor
	 */
	private $content;

	/**
	 * @var array
	 */
	private $attributes;

	/**
	 * @param string|HtmlArmor $content
	 * @param array $attributes
	 *
	 * @throws InvalidArgumentException
	 */
	public function __construct( $content, array $attributes = [] ) {
		Assert::parameterType( [ 'string', HtmlArmor::class ], $content, '$content' );

		$this->content = $content;
		$this->attributes = $attributes;
	}

	/**
	 * @return string HTML
	 */
	public function getContent() {
		return HtmlArmor::getHtml( $this->content );
	}

	/**
	 * @return array
	 */
	public function getAttributes() {
		return $this->attributes;
	}

	/**
	 * @return string HTML
	 */
	public function toHtml() {
		return Html::rawElement( 'td', $this->getAttributes(), $this->getContent() );
	}

}
