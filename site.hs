--------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings #-}
import           Data.Monoid (mappend)
import           System.FilePath (splitPath, joinPath)
import           Text.Pandoc.Options (
                                       readerExtensions
                                     , extensionsFromList
                                     , Extension ( Ext_tex_math_single_backslash, Ext_tex_math_double_backslash, Ext_tex_math_dollars, Ext_latex_macros )
                                     , writerHTMLMathMethod
                                     , HTMLMathMethod( MathJax )
                                     )
import           Hakyll


--------------------------------------------------------------------------------
main :: IO ()
main = hakyllWith config $ do
    match "images/*" $ do
        route   idRoute
        compile copyFileCompiler

    match "static/**" $ do
        route   rootRoute
        compile copyFileCompiler

    match "css/*" $ do
        route   idRoute
        compile compressCssCompiler

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


postCompiler :: Compiler (Item String)
postCompiler = pandocCompilerWith postReaderOptions postWriterOptions
    where
        postReaderOptions = defaultHakyllReaderOptions {
            readerExtensions = extensionsFromList
                [
                  Ext_tex_math_single_backslash  -- TeX math btw (..) [..]
                , Ext_tex_math_double_backslash  -- TeX math btw \(..\) \[..\]
                , Ext_tex_math_dollars           -- TeX math between $..$ or $$..$$
                , Ext_latex_macros               -- Parse LaTeX macro definitions (for math only)
                ]
        }
        postWriterOptions = defaultHakyllWriterOptions {
            writerHTMLMathMethod = MathJax ""
        }
    


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

