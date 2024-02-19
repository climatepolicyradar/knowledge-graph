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
wfLoadExtension( 'DeleteBatch' );
wfLoadExtension( 'Echo' );
wfLoadExtension( 'Nuke' );
wfLoadExtension( 'TemplateSandbox' );
wfLoadExtension( 'CodeEditor' );
wfLoadExtension( 'CodeMirror' );
wfLoadExtension( 'WikiEditor' );
wfLoadExtension( 'EmbedVideo' );
wfLoadExtension( 'Cite' );
wfLoadExtension( 'Graph' );
wfLoadExtension( 'Kartographer' );
wfLoadExtension( 'Math' );
wfLoadExtension( 'ParserFunctions' );
wfLoadExtension( 'Poem' );
wfLoadExtension( 'Score' );
wfLoadExtension( 'WikiHiero' );
wfLoadExtension( 'Scribunto' );
wfLoadExtension( 'SyntaxHighlight' );
wfLoadExtension( 'TemplateData' );
wfLoadExtension( 'ConfirmEdit' );
wfLoadExtension( 'ReCaptchaNoCaptcha' );
wfLoadExtension( 'TorBlock' );
wfLoadExtension( 'PageImages' );
wfLoadExtension( 'EntitySchema' );
wfLoadExtension( 'AdvancedSearch' );
wfLoadExtension( 'CLDR' );
wfLoadExtension( 'JsonConfig' );
wfLoadExtension( 'MobileFrontend' );
wfLoadExtension( 'MultimediaViewer' );
wfLoadExtension( 'OAuth' );
wfLoadExtension( 'RevisionSlider' );
wfLoadExtension( 'SecureLinkFixer' );
wfLoadExtension( 'Thanks' );
wfLoadExtension( 'TwoColConflict' );
wfLoadExtension( 'UniversalLanguageSelector' );
wfLoadExtension( 'Wikibase' );
wfLoadExtension( 'WikbaseInWikitext' );
wfLoadExtension( 'WikibaseManifest' );

# The following extensions are added because we have specific needs:
# we use WikibaseQualityConstraints to add inverse and symmetric constraints to our 
# properties, for hierarchical and related properties respectively
wfLoadExtension( 'WikibaseQualityConstraints' );

