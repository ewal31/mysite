--------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings #-}
import Data.Function (on)
import Data.Functor ((<&>))
import Hakyll.Process
import Data.List (groupBy)
import System.FilePath (joinPath, splitPath, takeDirectory, takeBaseName, takeExtension)
import Text.Pandoc.Highlighting (Style, kate, styleToCss)
import Text.Pandoc.Options (
                             extensionsFromList
                           , enableExtension
                           , disableExtension
                           , pandocExtensions
                           , Extension (
                                 Ext_backtick_code_blocks
                               , Ext_citations
                               , Ext_fenced_code_blocks
                               , Ext_footnotes
                               , Ext_inline_code_attributes
                               , Ext_latex_macros
                               , Ext_link_attributes
                               --, Ext_markdown_in_html_blocks
                               , Ext_raw_attribute
                               , Ext_raw_tex
                               , Ext_smart
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
-- import Text.Pandoc.Extensions (pandocExtensions)
import Hakyll.Images (Image, loadImage, scaleImageCompiler)
import Hakyll

jsToCompress :: Pattern
jsToCompress = fromGlob "static/**.js" .&&. (complement . fromGlob $ "static/**.min.js")

main :: IO ()
main = hakyllWith config $ do
    -- Javascript
    match (fromGlob "static/**" .&&. complement jsToCompress) $ do
        route   rootRoute
        compile copyFileCompiler

    match jsToCompress $ do
        route $ rootRoute `composeRoutes` setExtension "min.js"
        compile $ execCompilerWith (execName "terser") [HakFilePath, ProcArg "--compress"] CStdOut

    match (fromGlob "static/img/photography/**") $ version "compressed" $ do
        route $ customRoute getCompressedPhotoUrl
        compile $ loadImage
            >>= scaleImageCompiler 400 400

    -- CSS
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

    -- Posts
    match "posts/*" $ do
        route $ setExtension "html"
        compile $ postCompiler
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls

    create ["blog.html"] $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let archiveCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    constField "title" "Blog"                `mappend`
                    defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/blog.html" archiveCtx
                >>= loadAndApplyTemplate "templates/default.html" archiveCtx
                >>= relativizeUrls

    -- Notes
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

    -- Tools
    match "tools/**" $ do
        route $ setExtension "html"
        compile $ postCompiler
            >>= loadAndApplyTemplate "templates/note.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls

    create ["tools.html"] $ do
        route idRoute
        compile $ do
            tools <- loadAll "tools/**"

            let toolsCtx =
                    listField "tools" defaultContext (return tools) `mappend`
                    constField "title" "Tools"                      `mappend`
                    defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/tools.html" toolsCtx
                >>= loadAndApplyTemplate "templates/default.html" toolsCtx
                >>= relativizeUrls

    -- Presentations
    match "presentations/**" $ do
        route $ setExtension "html"
        compile $ presentationCompiler
            >>= loadAndApplyTemplate "templates/presentation.html" postCtx
            >>= relativizeUrls

    -- Photography
    create ["photography.html"] $ do
        route idRoute
        compile $ do
            let photoDir = "static/img/photography/**"
            photos <- loadAll (photoDir .&&. hasVersion "compressed") >>= mapM buildPhotoHtml . zip [1..]
            lightboxes <- loadAll (photoDir .&&. hasNoVersion) >>= mapM buildLightboxHtml . zip [1..]

            let photoCtx =
                    listField "photos" defaultContext (return photos) `mappend`
                    listField "lightboxes" defaultContext (return lightboxes) `mappend`
                    constField "title" "Photography" `mappend`
                    constField "css_file" "/css/gallery.css" `mappend`
                    defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/gallery.html" photoCtx
                >>= loadAndApplyTemplate "templates/default.html" photoCtx
                >>= relativizeUrls

    -- Homepage
    match "index.html" $ do
        route idRoute
        compile $ do
            let indexCtx =
                    constField "title" "Home" `mappend`
                    defaultContext
            getResourceBody
                >>= applyAsTemplate indexCtx
                >>= loadAndApplyTemplate "templates/default.html" indexCtx
                >>= relativizeUrls

    match "templates/*" $ compile templateCompiler

dropDirectory :: [String] -> [String]
dropDirectory []       = []
dropDirectory ("/":ds) = dropDirectory ds
dropDirectory ds       = tail ds

getCompressedPhotoUrl :: Identifier -> FilePath
getCompressedPhotoUrl = joinPath . addComp . dropDirectory . splitPath . toFilePath
    where addComp (x : xs) = x : "compressed" : xs

buildGalleryHtmlHelper :: Identifier -> Int -> FilePath -> Compiler (Item String)
buildGalleryHtmlHelper template idx filePath = makeItem ""
    >>= loadAndApplyTemplate template context
    where context = constField "name" name `mappend`
                    constField "file" (name ++ extension) `mappend`
                    constField "path" filePath `mappend`
                    constField "href" href `mappend`
                    constField "id" photoId `mappend`
                    defaultContext
          name = takeBaseName filePath
          extension = takeExtension filePath
          href = "#portfolio-item-" ++ show idx
          photoId = tail href

buildPhotoHtml :: (Int, Item Image) -> Compiler (Item String)
buildPhotoHtml (idx, photo) = buildGalleryHtmlHelper "templates/photo.html" idx filePath
    where filePath = getCompressedPhotoUrl . itemIdentifier $ photo

buildLightboxHtml :: (Int, Item CopyFile) -> Compiler (Item String)
buildLightboxHtml (idx, photo) = buildGalleryHtmlHelper "templates/lightbox.html" idx filePath
    where filePath = joinPath . dropDirectory . splitPath . toFilePath . itemIdentifier $ photo

rootRoute :: Routes
rootRoute = customRoute (joinPath . dropDirectory . splitPath . toFilePath)

postCompiler :: Compiler (Item String)
postCompiler = do
    csl <- load (fromFilePath "assets/csl/chicago.csl")
    bib <- load (fromFilePath "assets/bibliography/General.bib")
    writePandocWith postWriterOptions <$> (getResourceBody >>= readPandocBiblio postReaderOptions csl bib
                                                           >>= traverse (return . usingSideNotes))
    where
        -- TODO can probably delete this, the defaults look good
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
                , Ext_raw_attribute
                --, Ext_markdown_in_html_blocks
                ]
        }
        postWriterOptions = defaultHakyllWriterOptions {
            writerHTMLMathMethod = MathJax ""
          , writerHighlightStyle = Just pandocCodeStyle
        }

presentationCompiler :: Compiler (Item String)
presentationCompiler = do
    pandocCompilerWith presentationReaderOptions presentationWriterOptions
    where
        presentationReaderOptions = defaultHakyllReaderOptions {
            readerExtensions = disableExtension Ext_raw_tex pandocExtensions
        }
        presentationWriterOptions = defaultHakyllWriterOptions {
            writerHTMLMathMethod = MathJax ""
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
