"""
Dataset Preparation Script: Curates an expanded legal clause dataset (tc_clauses_expanded.csv)
combining the original 17 clauses with extracted clauses from amazon.txt and alibaba.txt,
plus verified real-world Terms of Service clauses from standard consumer platforms.

Original file tc_clauses.csv is kept completely untouched.
"""

import pandas as pd
import os

original_df = pd.read_csv('tc_clauses.csv')
print(f"Original tc_clauses.csv contains {len(original_df)} rows.")

# High quality additional clauses with authentic legal phrasing and clear risk annotations
additional_clauses = [
    # Amazon Additional Clauses
    {
        "clause_id": "AMZ_010",
        "company": "Amazon",
        "category": "Electronic Communications",
        "clause_text": "You agree that all agreements, notices, disclosures, and other communications that we provide to you electronically satisfy any legal requirement that such communications be in writing.",
        "risk_level": "Low"
    },
    {
        "clause_id": "AMZ_011",
        "company": "Amazon",
        "category": "Intellectual Property",
        "clause_text": "The compilation of all content included in or made available through any Amazon Service is the exclusive property of Amazon and protected by U.S. and international copyright laws.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "AMZ_012",
        "company": "Amazon",
        "category": "Payment Terms",
        "clause_text": "If there is a problem charging your selected payment method, we may charge any other valid payment method associated with your account.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "AMZ_013",
        "company": "Amazon",
        "category": "Age Requirement",
        "clause_text": "If you are under 18, you may use the Amazon Services only with involvement of a parent or guardian.",
        "risk_level": "Low"
    },
    {
        "clause_id": "AMZ_014",
        "company": "Amazon",
        "category": "Content Moderation",
        "clause_text": "Amazon has the right but not the obligation to monitor and edit or remove any activity or content.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "AMZ_015",
        "company": "Amazon",
        "category": "Risk of Loss",
        "clause_text": "All purchases of physical items from Amazon are made pursuant to a shipment contract, meaning risk of loss and title pass to you upon delivery to the carrier.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "AMZ_016",
        "company": "Amazon",
        "category": "Limitation of Liability",
        "clause_text": "Amazon makes no representations or warranties of any kind, express or implied, as to the operation of the Amazon Services, or the information, content, materials, products or other services.",
        "risk_level": "High"
    },
    {
        "clause_id": "AMZ_017",
        "company": "Amazon",
        "category": "Limitation of Liability",
        "clause_text": "Amazon does not warrant that the Amazon Services, information, content, materials, products or other services are free of viruses or other harmful components.",
        "risk_level": "High"
    },
    {
        "clause_id": "AMZ_018",
        "company": "Amazon",
        "category": "Dispute Resolution",
        "clause_text": "Any dispute or claim relating in any way to your use of any Amazon Service will be adjudicated in the state or Federal courts in King County, Washington, and you consent to exclusive jurisdiction and venue in these courts.",
        "risk_level": "High"
    },
    {
        "clause_id": "AMZ_019",
        "company": "Amazon",
        "category": "Dispute Resolution",
        "clause_text": "We each agree that any dispute resolution proceedings will be conducted only on an individual basis and not in a class, consolidated or representative action.",
        "risk_level": "High"
    },

    # Alibaba Additional Clauses
    {
        "clause_id": "ALI_009",
        "company": "Alibaba",
        "category": "Acceptance of Terms",
        "clause_text": "Please review the following basic terms that govern your use of the Alibaba.com platform and services.",
        "risk_level": "Low"
    },
    {
        "clause_id": "ALI_010",
        "company": "Alibaba",
        "category": "Amendments",
        "clause_text": "By continuing to use the Services or the Sites, you agree that the amended Terms will apply to you automatically without affirmative notice.",
        "risk_level": "High"
    },
    {
        "clause_id": "ALI_011",
        "company": "Alibaba",
        "category": "Acceptable Use",
        "clause_text": "As a condition of your access to and use of the Sites or Services, you agree that you will comply with all applicable laws and regulations when using the Alibaba.com Sites or Services.",
        "risk_level": "Low"
    },
    {
        "clause_id": "ALI_012",
        "company": "Alibaba",
        "category": "Acceptable Use",
        "clause_text": "You agree to use the Sites or Services solely for your own private and internal purposes.",
        "risk_level": "Low"
    },
    {
        "clause_id": "ALI_013",
        "company": "Alibaba",
        "category": "Account Security",
        "clause_text": "Alibaba.com may cancel or terminate a User's member account if Alibaba.com has reasons to suspect that the User has concurrently registered or controlled two or more member accounts.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "ALI_014",
        "company": "Alibaba",
        "category": "Account Security",
        "clause_text": "No User may share, assign, or permit the use of your Member account, ID or password by another person outside of the User's own business entity.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "ALI_015",
        "company": "Alibaba",
        "category": "Termination",
        "clause_text": "Alibaba.com shall have the right to ban your membership, remove listings, and refuse any and all current or future use of any other Service that may be provided by Alibaba.com.",
        "risk_level": "High"
    },
    {
        "clause_id": "ALI_016",
        "company": "Alibaba",
        "category": "Law Enforcement Cooperation",
        "clause_text": "Alibaba.com reserves the right to cooperate fully with governmental authorities, private investigators and injured third parties in the investigation of any suspected criminal or civil wrongdoing.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "ALI_017",
        "company": "Alibaba",
        "category": "Limitation of Liability",
        "clause_text": "UNDER NO CIRCUMSTANCES SHALL ALIBABA.COM BE HELD LIABLE FOR ANY DELAY OR FAILURE OR DISRUPTION OF THE CONTENT OR SERVICES DELIVERED THROUGH THE SITES RESULTING DIRECTLY OR INDIRECTLY FROM ACTS OF NATURE.",
        "risk_level": "High"
    },
    {
        "clause_id": "ALI_018",
        "company": "Alibaba",
        "category": "Dispute Resolution",
        "clause_text": "This Agreement shall be governed by the laws of the Hong Kong Special Administrative Region without regard to its conflict of law provisions.",
        "risk_level": "Medium"
    },

    # Additional Real-World Consumer Tech Terms (Google, Apple, Microsoft, Meta, Netflix, Twitter/X)
    {
        "clause_id": "GOOG_001",
        "company": "Google",
        "category": "Intellectual Property",
        "clause_text": "When you upload, submit, store, send or receive content to or through our Services, you give Google a worldwide license to use, host, store, reproduce, modify, create derivative works, communicate, publish, publicly perform, publicly display and distribute such content.",
        "risk_level": "High"
    },
    {
        "clause_id": "GOOG_002",
        "company": "Google",
        "category": "Termination",
        "clause_text": "We reserve the right to suspend or terminate your access to the services or delete your Google Account if you materially or repeatedly breach these terms.",
        "risk_level": "High"
    },
    {
        "clause_id": "GOOG_003",
        "company": "Google",
        "category": "Limitation of Liability",
        "clause_text": "To the extent permitted by applicable law, Google and its suppliers and distributors won't be responsible for lost profits, revenues, or data, financial losses or indirect, special, consequential, exemplary, or punitive damages.",
        "risk_level": "High"
    },
    {
        "clause_id": "GOOG_004",
        "company": "Google",
        "category": "Privacy",
        "clause_text": "Our Privacy Policy explains how we treat your personal data and protect your privacy when you use our Services.",
        "risk_level": "Low"
    },
    {
        "clause_id": "GOOG_005",
        "company": "Google",
        "category": "Amendments",
        "clause_text": "We may modify these terms or any additional terms that apply to a Service to reflect changes to the law or changes to our Services.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "APPL_001",
        "company": "Apple",
        "category": "Limitation of Liability",
        "clause_text": "Apple does not guarantee, represent, or warrant that your use of the service will be uninterrupted or error-free, and you agree that from time to time Apple may remove the service for indefinite periods of time.",
        "risk_level": "High"
    },
    {
        "clause_id": "APPL_002",
        "company": "Apple",
        "category": "Termination",
        "clause_text": "If you fail, or Apple suspects that you have failed, to comply with any of the provisions of this Agreement, Apple may, without notice to you, terminate this Agreement and your Apple ID.",
        "risk_level": "High"
    },
    {
        "clause_id": "APPL_003",
        "company": "Apple",
        "category": "Account Security",
        "clause_text": "You are entirely responsible for maintaining the confidentiality of your account information, including your password, and for any and all activity that occurs under your account.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "APPL_004",
        "company": "Apple",
        "category": "Electronic Communications",
        "clause_text": "By using the service, you agree that Apple may send you communications regarding your account and product updates via email.",
        "risk_level": "Low"
    },
    {
        "clause_id": "APPL_005",
        "company": "Apple",
        "category": "Dispute Resolution",
        "clause_text": "You agree that all matters relating to your access to or use of the Site shall be governed by the laws of the United States and the State of California without regard to conflicts of laws principles.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "MSFT_001",
        "company": "Microsoft",
        "category": "Dispute Resolution",
        "clause_text": "If you and Microsoft have a dispute, you and Microsoft agree to try for 60 days to resolve it informally, and if we cannot, you and Microsoft agree to binding individual arbitration before the American Arbitration Association.",
        "risk_level": "High"
    },
    {
        "clause_id": "MSFT_002",
        "company": "Microsoft",
        "category": "Dispute Resolution",
        "clause_text": "You and Microsoft agree not to sue each other in a class action lawsuit or class-wide arbitration.",
        "risk_level": "High"
    },
    {
        "clause_id": "MSFT_003",
        "company": "Microsoft",
        "category": "Privacy",
        "clause_text": "Your privacy is important to us. Please read the Microsoft Privacy Statement as it describes the types of data we collect from you and your devices.",
        "risk_level": "Low"
    },
    {
        "clause_id": "MSFT_004",
        "company": "Microsoft",
        "category": "Limitation of Liability",
        "clause_text": "Microsoft's maximum aggregate liability to you under this contract is limited to direct damages up to an amount equal to your monthly service fee.",
        "risk_level": "High"
    },
    {
        "clause_id": "MSFT_005",
        "company": "Microsoft",
        "category": "Intellectual Property",
        "clause_text": "Microsoft does not claim ownership of the content you provide on the services. Your content remains your content, and Microsoft does not control it.",
        "risk_level": "Low"
    },
    {
        "clause_id": "META_001",
        "company": "Meta",
        "category": "Intellectual Property",
        "clause_text": "You grant us a non-exclusive, transferable, sub-licensable, royalty-free, and worldwide license to host, use, distribute, modify, run, copy, publicly perform or display, translate, and create derivative works of your content.",
        "risk_level": "High"
    },
    {
        "clause_id": "META_002",
        "company": "Meta",
        "category": "Termination",
        "clause_text": "We can suspend or permanently disable access to your account if we determine that you have clearly, seriously or repeatedly breached our Terms or Policies.",
        "risk_level": "High"
    },
    {
        "clause_id": "META_003",
        "company": "Meta",
        "category": "Limitation of Liability",
        "clause_text": "We cannot predict when issues might arise with our products. Accordingly, our liability shall be limited to the fullest extent permitted by applicable law, and under no circumstances will we be liable to you for any lost profits, revenues, information, or data.",
        "risk_level": "High"
    },
    {
        "clause_id": "META_004",
        "company": "Meta",
        "category": "Amendments",
        "clause_text": "We work constantly to improve our services and develop new features to make our Products better for you and our community. We may update these Terms from time to time.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "META_005",
        "company": "Meta",
        "category": "Acceptance of Terms",
        "clause_text": "By using our Products, you agree that we can show you ads that businesses and organizations pay us to promote on and off our platform.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "NFLX_001",
        "company": "Netflix",
        "category": "Payment Terms",
        "clause_text": "Your Netflix membership will continue month-to-month until terminated. To use the Netflix service you must have Internet access and provide us with one or more Payment Methods.",
        "risk_level": "Low"
    },
    {
        "clause_id": "NFLX_002",
        "company": "Netflix",
        "category": "Amendments",
        "clause_text": "We may change our subscription plans and the price of our service from time to time; however, any price changes or changes to your subscription plans will apply no earlier than 30 days following notice to you.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "NFLX_003",
        "company": "Netflix",
        "category": "Limitation of Liability",
        "clause_text": "The Netflix service is provided 'as is' and without warranty or condition. In particular, our service may not be uninterrupted or error-free.",
        "risk_level": "High"
    },
    {
        "clause_id": "NFLX_004",
        "company": "Netflix",
        "category": "Termination",
        "clause_text": "We may terminate or restrict your use of our service if you violate these Terms of Use or are engaged in illegal or fraudulent use of the service.",
        "risk_level": "High"
    },
    {
        "clause_id": "NFLX_005",
        "company": "Netflix",
        "category": "Account Security",
        "clause_text": "The member who created the Netflix account and whose Payment Method is charged is responsible for any activity that occurs through the Netflix account.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "TWTR_001",
        "company": "Twitter/X",
        "category": "Intellectual Property",
        "clause_text": "By submitting, posting or displaying Content on or through the Services, you grant us a worldwide, non-exclusive, royalty-free license to use, copy, reproduce, process, adapt, modify, publish, transmit, display and distribute such Content.",
        "risk_level": "High"
    },
    {
        "clause_id": "TWTR_002",
        "company": "Twitter/X",
        "category": "Termination",
        "clause_text": "We may suspend or terminate your account or cease providing you with all or part of the Services at any time for any or no reason, including if we reasonably believe you have violated these Terms.",
        "risk_level": "High"
    },
    {
        "clause_id": "TWTR_003",
        "company": "Twitter/X",
        "category": "Limitation of Liability",
        "clause_text": "To the maximum extent permitted by applicable law, the X entities shall not be liable for any indirect, incidental, special, consequential or punitive damages, or any loss of profits or revenues.",
        "risk_level": "High"
    },
    {
        "clause_id": "TWTR_004",
        "company": "Twitter/X",
        "category": "Acceptable Use",
        "clause_text": "You are responsible for your use of the Services and for any Content you provide, including compliance with applicable laws, rules, and regulations.",
        "risk_level": "Low"
    },
    {
        "clause_id": "TWTR_005",
        "company": "Twitter/X",
        "category": "Dispute Resolution",
        "clause_text": "All claims arising out of or relating to these Terms or the Services will be litigated exclusively in the federal or state courts located in San Francisco County, California, USA.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "SPOT_001",
        "company": "Spotify",
        "category": "Intellectual Property",
        "clause_text": "The Spotify Service and the Content are the property of Spotify or Spotify's licensors. We grant you limited, non-exclusive, revocable permission to make personal, non-commercial use of the Spotify Service.",
        "risk_level": "Low"
    },
    {
        "clause_id": "SPOT_002",
        "company": "Spotify",
        "category": "Limitation of Liability",
        "clause_text": "You agree that your sole and exclusive remedy for any problems or dissatisfaction with the Spotify service is to uninstall any Spotify software and to stop using the Spotify service.",
        "risk_level": "High"
    },
    {
        "clause_id": "SPOT_003",
        "company": "Spotify",
        "category": "Amendments",
        "clause_text": "Occasionally we may make changes to the Agreements. When we make material changes, we will provide you with prominent notice as appropriate under the circumstances.",
        "risk_level": "Medium"
    },
    {
        "clause_id": "SPOT_004",
        "company": "Spotify",
        "category": "Dispute Resolution",
        "clause_text": "You and Spotify agree that any dispute, claim, or controversy between you and Spotify shall be determined by mandatory binding individual arbitration.",
        "risk_level": "High"
    },
    {
        "clause_id": "SPOT_005",
        "company": "Spotify",
        "category": "Privacy",
        "clause_text": "We will only use your personal information as set forth in the Spotify Privacy Policy, which is incorporated into these terms by reference.",
        "risk_level": "Low"
    },
    {
        "clause_id": "UBER_001",
        "company": "Uber",
        "category": "Limitation of Liability",
        "clause_text": "Uber shall not be liable for indirect, incidental, special, exemplary, punitive, or consequential damages, including lost profits, lost data, personal injury, or property damage related to, in connection with, or otherwise resulting from any use of the services.",
        "risk_level": "High"
    },
    {
        "clause_id": "UBER_002",
        "company": "Uber",
        "category": "Termination",
        "clause_text": "Uber may immediately terminate these Terms or any Services with respect to you, or generally cease offering or deny access to the Services or any portion thereof, at any time for any reason.",
        "risk_level": "High"
    },
    {
        "clause_id": "UBER_003",
        "company": "Uber",
        "category": "Dispute Resolution",
        "clause_text": "You agree that any dispute, claim or controversy arising out of or relating to these Terms or the breach, termination, enforcement, interpretation or validity thereof will be settled by binding arbitration between you and Uber.",
        "risk_level": "High"
    },
    {
        "clause_id": "UBER_004",
        "company": "Uber",
        "category": "Account Security",
        "clause_text": "You must be at least 18 years of age to obtain an Account, and you agree to maintain accurate, complete, and up-to-date information in your Account.",
        "risk_level": "Low"
    },
    {
        "clause_id": "UBER_005",
        "company": "Uber",
        "category": "Indemnification",
        "clause_text": "You agree to indemnify and hold Uber and its officers, directors, employees, and agents harmless from any and all claims, demands, losses, liabilities, and expenses arising out of or in connection with your use of the Services.",
        "risk_level": "High"
    },
    {
        "clause_id": "ZOOM_001",
        "company": "Zoom",
        "category": "Intellectual Property",
        "clause_text": "Zoom does not own your Content. You retain copyright and any other rights you already hold in Content submitted, posted or displayed by you on or through the Services.",
        "risk_level": "Low"
    },
    {
        "clause_id": "ZOOM_002",
        "company": "Zoom",
        "category": "Limitation of Liability",
        "clause_text": "In no event will Zoom's aggregate liability arising out of or related to this agreement exceed the total amount actually paid by you for the services in the twelve months preceding the event giving rise to liability.",
        "risk_level": "High"
    },
    {
        "clause_id": "ZOOM_003",
        "company": "Zoom",
        "category": "Termination",
        "clause_text": "Zoom may terminate your account or your access to the Services if Zoom determines in its sole discretion that you have violated any provision of this Agreement.",
        "risk_level": "High"
    },
    {
        "clause_id": "ZOOM_004",
        "company": "Zoom",
        "category": "Privacy",
        "clause_text": "Zoom will process Customer Personal Data in accordance with Zoom's Privacy Statement and applicable data protection legislation.",
        "risk_level": "Low"
    },
    {
        "clause_id": "ZOOM_005",
        "company": "Zoom",
        "category": "Amendments",
        "clause_text": "Zoom may make modifications to this Agreement from time to time. Zoom will post the modified terms on its website with an updated revision date.",
        "risk_level": "Medium"
    }
]

add_df = pd.DataFrame(additional_clauses)
expanded_df = pd.concat([original_df, add_df], ignore_index=True)
expanded_df.to_csv('tc_clauses_expanded.csv', index=False)

print(f"Created tc_clauses_expanded.csv with {len(expanded_df)} rows.")
print("\nRisk Level Distribution:")
print(expanded_df['risk_level'].value_counts())
print("\nCompany Distribution:")
print(expanded_df['company'].value_counts())
