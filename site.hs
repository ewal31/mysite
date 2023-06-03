--------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings #-}
import           Control.Monad (liftM)
import           Data.Monoid (mappend)
import           System.FilePath (splitPath, joinPath)
import           Text.Pandoc.Highlighting (Style, kate, styleToCss)
import           Text.Pandoc.Options (
                                       extensionsFromList
									 , Extension (
									       Ext_backtick_code_blocks
									     , Ext_citations
										 , Ext_fenced_code_blocks
										 , Ext_tex_math_single_backslash
										 , Ext_tex_math_double_backslash
										 , Ext_tex_math_dollars
										 , Ext_latex_macros
										 , Ext_inline_code_attributes
										 , Ext_link_attributes
									   )
									 , HTMLMathMethod( MathJax )
									 , readerExtensions
                                     , writerHighlightStyle
									 , writerHTMLMathMethod
                                     )
import           Hakyll


--------------------------------------------------------------------------------
main :: IO ()
main = hakyllWith config $ do
    --match "images/*" $ do
    --    route   idRoute
    --    compile copyFileCompiler

    match "static/**" $ do
        route   rootRoute
        compile copyFileCompiler

    match "css/*" $ do
        route   idRoute
        compile compressCssCompiler
		
    create ["css/syntax.css"] $ do
      route idRoute
      compile $ do
        makeItem $ styleToCss pandocCodeStyle
		
    -- Bibtex entries (for bibliography)
    match "assets/bibliography/*" $ compile biblioCompiler
    match "assets/csl/*" $ compile cslCompiler

    -- match (fromList ["about.rst", "contact.markdown"]) $ do
    --     route   $ setExtension "html"
    --     compile $ pandocCompiler
    --         >>= loadAndApplyTemplate "templates/default.html" defaultContext
    --         >>= relativizeUrls

    match "posts/*" $ do
        route $ setExtension "html"
        compile $ postCompiler
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls

    create ["archive.html"] $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let archiveCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    constField "title" "Archives"            `mappend`
                    defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/archive.html" archiveCtx
                >>= loadAndApplyTemplate "templates/default.html" archiveCtx
                >>= relativizeUrls


    match "index.html" $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let indexCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    constField "title" "Home"                `mappend`
                    defaultContext
            getResourceBody
                >>= applyAsTemplate indexCtx
                >>= loadAndApplyTemplate "templates/default.html" indexCtx
                >>= relativizeUrls

    match "templates/*" $ compile templateCompiler


rootRoute :: Routes
rootRoute = customRoute (joinPath . dropDirectory . splitPath . toFilePath)
    where
        dropDirectory []       = []
        dropDirectory ("/":ds) = dropDirectory ds
        dropDirectory ds       = tail ds


-- TODO need to modify this to only include what is necessary for the page
postCompiler :: Compiler (Item String)
postCompiler = do --pandocCompilerWith postReaderOptions postWriterOptions
	csl <- load (fromFilePath "assets/csl/chicago.csl")
	bib <- load (fromFilePath "assets/bibliography/General.bib")
	liftM (writePandocWith postWriterOptions) (getResourceBody >>= readPandocBiblio postReaderOptions csl bib)
    where
        postReaderOptions = defaultHakyllReaderOptions {
            readerExtensions = extensionsFromList
                [ -- https://hackage.haskell.org/package/pandoc-3.1.2/docs/Text-Pandoc-Extensions.html#t:Extension
				  Ext_backtick_code_blocks       -- GitHub style ``` code blocks
				, Ext_citations                  -- Pandoc/citeproc citations
				, Ext_fenced_code_blocks         -- Parse fenced code blocks
                , Ext_tex_math_single_backslash  -- TeX math btw (..) [..]
                , Ext_tex_math_double_backslash  -- TeX math btw \(..\) \[..\]
                , Ext_tex_math_dollars           -- TeX math between $..$ or $$..$$
                , Ext_latex_macros               -- Parse LaTeX macro definitions (for math only)
				, Ext_inline_code_attributes     -- Allow attributes on inline code
				, Ext_link_attributes            -- link and image attributes
                ]
        }
        postWriterOptions = defaultHakyllWriterOptions {
            writerHTMLMathMethod = MathJax ""
		  , writerHighlightStyle = Just pandocCodeStyle
        }

-- Styles for code highlighting
-- https://hackage.haskell.org/package/pandoc-3.1.2/docs/Text-Pandoc-Highlighting.html
pandocCodeStyle :: Style
pandocCodeStyle = kate

config :: Configuration
config = defaultConfiguration {
        destinationDirectory = "docs",
        previewHost = "0.0.0.0"
    }

--------------------------------------------------------------------------------
postCtx :: Context String
postCtx =
    dateField "date" "%B %e, %Y" `mappend`
    defaultContext

