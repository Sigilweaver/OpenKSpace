import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
    title: 'OpenKSpace',
    tagline: 'Pure-Rust Cartesian MRI k-space reconstruction from ISMRMRD files',
    favicon: 'img/favicon.ico',

    markdown: {
        mermaid: true,
        hooks: {
            onBrokenMarkdownLinks: 'warn',
        },
    },
    plugins: ['docusaurus-plugin-llms-txt'],
    themes: ['@docusaurus/theme-mermaid'],

    url: 'https://sigilweaver.app',
    baseUrl: '/openkspace/docs/',

    organizationName: 'Sigilweaver',
    projectName: 'OpenKSpace',

    onBrokenLinks: 'throw',

    i18n: {
        defaultLocale: 'en',
        locales: ['en'],
    },

    presets: [
        [
            'classic',
            {
                docs: {
                    routeBasePath: '/',
                    sidebarPath: './sidebars.ts',
                    editUrl: 'https://github.com/Sigilweaver/OpenKSpace/tree/main/docs/',
                },
                blog: false,
                sitemap: {
                    changefreq: 'weekly',
                    priority: 0.5,
                    filename: 'sitemap.xml',
                },
                theme: {
                    customCss: './src/css/custom.css',
                },
            } satisfies Preset.Options,
        ],
    ],

    themeConfig: {
        metadata: [
            { name: 'keywords', content: 'OpenKSpace, MRI, k-space, ISMRMRD, GRAPPA, SENSE, ESPIRiT, compressed sensing, reconstruction, Rust' },
            { name: 'description', content: 'OpenKSpace is a pure-Rust library and CLI for Cartesian MRI k-space reconstruction from ISMRMRD .h5 files.' },
        ],
        colorMode: {
            defaultMode: 'dark',
            disableSwitch: false,
            respectPrefersColorScheme: true,
        },
        navbar: {
            title: 'Sigilweaver',
            logo: {
                alt: 'Sigilweaver logo',
                src: 'img/logo.svg',
                href: 'https://sigilweaver.app',
                target: '_self',
            },
            items: [
                {
                    type: 'dropdown',
                    label: 'Projects',
                    position: 'left',
                    items: [
                        { label: 'OpenKSpace', href: 'https://sigilweaver.app/openkspace/docs/' },
                        { label: 'BioLance', href: 'https://sigilweaver.app/biolance/docs/' },
                        { label: 'DICOM-Atlas', href: 'https://sigilweaver.app/dicom-atlas/docs/' },
                        { label: 'OpenMassSpec', href: 'https://sigilweaver.app/openmassspec/docs/' },
                        { label: 'All projects', href: 'https://sigilweaver.app/docs/' },
                    ],
                },
                {
                    href: 'https://github.com/Sigilweaver/OpenKSpace',
                    label: 'GitHub',
                    position: 'right',
                },
            ],
        },
        footer: {
            style: 'dark',
            links: [
                {
                    title: 'Project',
                    items: [
                        { label: 'GitHub', href: 'https://github.com/Sigilweaver/OpenKSpace' },
                        { label: 'Issues', href: 'https://github.com/Sigilweaver/OpenKSpace/issues' },
                        { label: 'crates.io', href: 'https://crates.io/crates/openkspace-cli' },
                    ],
                },
                {
                    title: 'Related',
                    items: [
                        { label: 'BioLance', href: 'https://sigilweaver.app/biolance/docs/' },
                        { label: 'DICOM-Atlas', href: 'https://sigilweaver.app/dicom-atlas/docs/' },
                        { label: 'All projects', href: 'https://sigilweaver.app/docs/' },
                    ],
                },
                {
                    title: 'Legal',
                    items: [
                        { label: 'Terms of Use', href: 'https://sigilweaver.app/terms' },
                        { label: 'Privacy Policy', href: 'https://sigilweaver.app/privacy' },
                    ],
                },
            ],
            copyright: `Copyright ${new Date().getFullYear()} Sigilweaver Holdings LLC. OpenKSpace is Apache-2.0 licensed. Documentation licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener noreferrer">CC-BY-SA 4.0</a>.`,
        },
        prism: {
            theme: prismThemes.github,
            darkTheme: prismThemes.dracula,
            additionalLanguages: ['rust', 'toml', 'bash'],
        },
    } satisfies Preset.ThemeConfig,
};

export default config;
