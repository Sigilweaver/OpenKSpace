import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
    docsSidebar: [
        'intro',
        {
            type: 'category',
            label: 'Getting started',
            collapsed: false,
            items: [
                'install',
                'quickstart-cli',
                'quickstart-rust',
            ],
        },
        {
            type: 'category',
            label: 'Pipeline',
            items: [
                'calibration',
                'reconstruction',
                'output',
            ],
        },
        {
            type: 'category',
            label: 'Reference',
            items: [
                'cli',
                'citations',
                'roadmap',
            ],
        },
    ],
};

export default sidebars;
