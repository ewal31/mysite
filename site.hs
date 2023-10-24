--------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings #-}
import Data.Function (on)
import Data.Functor ((<&>))
import Data.List (groupBy)
import System.FilePath (joinPath, splitPath, takeDirectory)
import Text.Pandoc.Highlighting (Style, kate, styleToCss)
import Text.Pandoc.Options (
                             extensionsFromList
                           , Extension (
                                 Ext_backtick_code_blocks
                               , Ext_citations
                               , Ext_fenced_code_blocks
                               , Ext_footnotes
                               , Ext_inline_code_attributes
                               , Ext_latex_macros
                               , Ext_link_attributes
                               --, Ext_markdown_in_html_blocks
                               --, Ext_raw_attribute
                               , Ext_tex_math_dollars
                               , Ext_tex_math_double_backslash
                               , Ext_tex_math_single_backslash
                             )
                           , HTMLMathMethod( MathJax )
                           , readerExtensions
                           , writerHighlightStyle
                           , writerHTMLMathMethod
                           )
import Text.Pandoc.SideNote (usingSideNotes)
import Hakyll

--------------------------------------------------------------------------------
main :: IO ()
main = hakyllWith config $ do
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
                    constField "title" "Archive"            `mappend`
                    defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/archive.html" archiveCtx
                >>= loadAndApplyTemplate "templates/default.html" archiveCtx
                >>= relativizeUrls

    match "notes/**" $ do
        route $ setExtension "html"
        compile $ postCompiler
            >>= loadAndApplyTemplate "templates/note.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls

    create ["notes.html"] $ do
        route idRoute
        compile $ do
            groupedNotes <- loadAll "notes/**" <&> groupBy (on (==) toDirectory)

            -- TODO don't want the urls to have spaces
            let groupCtx grp =
                    constField "notes-group" (joinPath . tail . splitPath . toDirectory . head $ grp) `mappend`
                    listField "notes" defaultContext (return grp)

            htmlGroupLists <- mapM (\grp -> makeItem (""::String) >>= loadAndApplyTemplate "templates/notes-list.html" (groupCtx grp)) groupedNotes

            let notesCtx =
                    listField "notes-list" defaultContext (return htmlGroupLists) `mappend`
                    constField "title" "Notes"                                    `mappend`
                    defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/notes-index.html" notesCtx
                >>= loadAndApplyTemplate "templates/default.html" notesCtx
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

-- csl <- load (fromFilePath "assets/csl/chicago.csl")
-- bib <- load (fromFilePath "assets/bibliography/General.bib")
-- liftM (writePandocWith postWriterOptions) (getResourceBody >>= readPandocBiblio postReaderOptions csl bib)
-- TODO need to modify this to only include what is necessary for the page
postCompiler :: Compiler (Item String)
postCompiler = do --pandocCompilerWith postReaderOptions postWriterOptions
    csl <- load (fromFilePath "assets/csl/chicago.csl")
    bib <- load (fromFilePath "assets/bibliography/General.bib")
    writePandocWith postWriterOptions <$> (getResourceBody >>= readPandocBiblio postReaderOptions csl bib
                                                           >>= traverse (return . usingSideNotes))
    --body <- getResourceBody
    --parsed <- (readPandocBiblio postReaderOptions csl bib body)
    --return (writePandocWith postWriterOptions parsed)
    where
        postReaderOptions = defaultHakyllReaderOptions {
            readerExtensions = extensionsFromList
                [ -- https://hackage.haskell.org/package/pandoc-3.1.2/docs/Text-Pandoc-Extensions.html#t:Extension
                  Ext_backtick_code_blocks       -- GitHub style ``` code blocks
                , Ext_citations                  -- Pandoc/citeproc citations
                , Ext_fenced_code_blocks         -- Parse fenced code blocks
                , Ext_footnotes
                , Ext_tex_math_single_backslash  -- TeX math btw (..) [..]
                , Ext_tex_math_double_backslash  -- TeX math btw \(..\) \[..\]
                , Ext_tex_math_dollars           -- TeX math between $..$ or $$..$$
                , Ext_latex_macros               -- Parse LaTeX macro definitions (for math only)
                , Ext_inline_code_attributes     -- Allow attributes on inline code
                , Ext_link_attributes            -- link and image attributes
                --, Ext_raw_attribute
                --, Ext_markdown_in_html_blocks
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

toDirectory :: Item a -> FilePath
toDirectory = takeDirectory . toFilePath . itemIdentifier
