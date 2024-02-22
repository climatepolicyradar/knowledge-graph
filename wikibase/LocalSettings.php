<?php

/*******************************/
/* Enable Federated properties */
/*******************************/
#$wgWBRepoSettings['federatedPropertiesEnabled'] = true;

/*******************************/
/* Enables ConfirmEdit Captcha */
/*******************************/
#wfLoadExtension( 'ConfirmEdit/QuestyCaptcha' );
#$wgCaptchaQuestions = [
#  'What animal' => 'dog',
#];

#$wgCaptchaTriggers['edit']          = true;
#$wgCaptchaTriggers['create']        = true;
#$wgCaptchaTriggers['createtalk']    = true;
#$wgCaptchaTriggers['addurl']        = true;
#$wgCaptchaTriggers['createaccount'] = true;
#$wgCaptchaTriggers['badlogin']      = true;

/*******************************/
/* Disable UI error-reporting  */
/*******************************/
#ini_set( 'display_errors', 0 );

# these are all of the standard wikibase extensions, see https://www.wbstack.com/users/wiki.html
wfLoadExtension( 'AdvancedSearch' );
wfLoadExtension( 'Babel' );
wfLoadExtension( 'Cite' );
wfLoadExtension( 'CLDR' );
wfLoadExtension( 'CodeEditor' );
wfLoadExtension( 'CodeMirror' );
wfLoadExtension( 'ConfirmEdit' );
wfLoadExtension( 'DeleteBatch' );
wfLoadExtension( 'Echo' );
wfLoadExtension( 'EntitySchema' );
wfLoadExtension( 'Graph' );
wfLoadExtension( 'JsonConfig' );
wfLoadExtension( 'Kartographer' );
wfLoadExtension( 'Math' );
wfLoadExtension( 'MobileFrontend' );
wfLoadExtension( 'MultimediaViewer' );
wfLoadExtension( 'Nuke' );
wfLoadExtension( 'OAuth' );
wfLoadExtension( 'PageImages' );
wfLoadExtension( 'ParserFunctions' );
wfLoadExtension( 'Poem' );
wfLoadExtension( 'RevisionSlider' );
wfLoadExtension( 'Score' );
wfLoadExtension( 'Scribunto' );
wfLoadExtension( 'SecureLinkFixer' );
wfLoadExtension( 'TemplateData' );
wfLoadExtension( 'TemplateSandbox' );
wfLoadExtension( 'Thanks' );
wfLoadExtension( 'TorBlock' );
wfLoadExtension( 'TwoColConflict' );
wfLoadExtension( 'UniversalLanguageSelector' );
wfLoadExtension( 'Wikibase' );
wfLoadExtension( 'WikibaseManifest' );
wfLoadExtension( 'WikiEditor' );
wfLoadExtension( 'WikiHiero' );

# We haven't been able to find/download these ones
// wfLoadExtension( 'EmbedVideo' );
// wfLoadExtension( 'ReCaptchaNoCaptcha' );
// wfLoadExtension( 'SyntaxHighlight' );
// wfLoadExtension( 'WikbaseInWikitext' );

# The following extensions are added because we have specific needs:
# we use WikibaseQualityConstraints to add inverse and symmetric constraints to our 
# properties, for hierarchical and related properties respectively
wfLoadExtension( 'WikibaseQualityConstraints' );

