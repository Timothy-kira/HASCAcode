# 3rd WEAR Dataset Challenge (HASCA 2026) — competition pages



---

## abstract

The **3rd WEAR Dataset Challenge** is a Human Activity Recognition prediction challenge based on the [WEAR dataset](http://mariusbock.github.io/wear/). The challenge will be part of the [HASCA Workshop](http://hasca2026.hasc.jp/) at [UbiComp/ ISWC 2026](https://www.ubicomp.org/ubicomp-iswc-2026/).

With previous iterations of the WEAR challenge focusing solely on inertial data, this year we want to challenge the wearable community to explore how to most effectively combine egocentric cameras with inertial sensors. 
To do so, this year, we provide random 1-second sliding windows from a single inertial sensor, as well as pre-extracted, frame-wise features ([VideoMAEv2-Base](https://huggingface.co/OpenGVLab/VideoMAEv2-Base)) from the egocentric camera's video stream.

⚠️ **Update – April 26, 2025:** A bug in the video feature generation has been fixed. Please redownload the test data. Sample ordering is unchanged; prior submissions without video features are unaffected.


---

## Description

The WEAR Challenge will again be hosted via Kaggle. Submission of prediction results will only be done via Kaggle. By using Kaggle's public and private leaderboard we aim to provide a more transparent and fair ranking process. Challenge participants will further be able to discuss the challenge and share their solutions on the Kaggle platform.

## IMPORTANT DISCLAIMER
**To be part of the final ranking and eligible for prices, participants will be required to submit a technical report to the HASCA workshop. The paper should contain technical description of the processing pipeline, the algorithms and the results achieved using the original WEAR dataset. An example of a technical report can be found here [winners of last year's challenge](https://doi.org/10.1145/3714394.3756194). Submissions must follow the HASCA format (see below for more details).**

## Other Important dates (same as HASCA)

*   **HASCA-WEAR paper submission**: July 5 9:00 AoE, 2026
*   **HASCA-WEAR review notification**: July 22, 2026
*   **HASCA-WEAR camera ready submission**: July 31, 2026
*   **HASCA Workshop**: October 11 or 12 (*tentative*), 2026 in Shanghai, China

## Submission Guidelines
### Final Predictions
Submission of the predictions will happen solely via Kaggle. All details on the submission format and final submissions can be found on here on Kaggle.

The final ranking of the WEAR Dataset challenge will be announced at the HASCA workhop. Ranking of the teams, which will be eligible for prizes, will be determined based on the private leaderboard score as of the technical report submission deadline. 

When submitting their technical report, participants will need to:
- include their team name as specified on Kaggle in the technical report, and
- make sure they selected the correct final submission on Kaggle to be used for ranking (see "Submissions" tab).
                
**NOTE:** Kaggle will automatically pick your best scoring public leaderboard solution, if you do not select anything in the "Submissions" tab. 

Prizes will be awarded to the three teams with the highest private leaderboard score at the time of the paper submission deadline.

### Technical Report
Submission of the technical report will happen via [PCS](https://new.precisionconference.com/submissions) *(select SIGCHI / UbiComp/ISWC 2026 / HASCA)*. Within the submission form please select "Submission type" -> "WEAR". Your technical report should detail your solution as well as provide preliminary results you achieved on the train dataset.  An example of a technical report can be found here [winners of last year's challenge](https://doi.org/10.1145/3714394.3756194). Please follow the formatting guidelines of the [HASCA workshop](https://hasca2026.hasc.jp/). 

The paper must be a maximum of 6 pages in the 2-column format. References do not count to the page limit, but all texts and figures/tables must be in the first 6 pages. Submissions do not need to be anonymous. All publications will be peer reviewed together with their contribution to the topic of the workshop. The accepted papers will be published in the UbiComp/ISWC 2025 adjunct proceedings, which will be included in the ACM Digital Library.

## About the WEAR Dataset

The WEAR dataset is an outdoor sports dataset for inertial- and video-based human activity recognition (HAR). The dataset comprises data from 22 participants performing a total of 18 different workout activities with untrimmed inertial (acceleration) and egocentric video data recorded at 11 different outside locations. WEAR provides a challenging prediction scenario marked by purposely introduced activity variations as well as an overall small information overlap across modalities. This challenge focuses on the inertial data only.


---

## Evaluation

Submissions are evaluated on [macro F1-Score](https://en.wikipedia.org/wiki/F-score#Macro_F1) between the predicted and the ground truth activity label of each sliding window in the test set. The macro F1-score is the arithmetic mean of class-wise F1 scores.

## Submission File
For each ID in the test set, you must predict the activity label for the `target_feature `. The file should contain a header and have the following format:

    id,target_feature
    1,0
    2,10
    3,2
    etc.

For more details see the [Data](https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026/data) section of this challenge


---

## data-description

⚠️ Update – April 26, 2025: A bug in the video feature generation has been fixed. Please redownload the test data. Sample ordering is unchanged; prior submissions without video features are unaffected.

## Files

*   **train** – raw inertial data and frame-wise VideoMAEv2-Base features of the WEAR dataset, split by participant. Filenames ending in *_2.csv indicate a repeated recording (see [paper](https://arxiv.org/abs/2304.05088v4)). Note the difference in sampling rates: inertial = 50 Hz, video = 30 FPS.
* **test** – all test split files (see below for details)
* `sample_submission.csv` – a sample submission file in the correct format
* `participant_meta.txt` – supplemental metadata about participants and recording conditions of the test set (for train set info, see the [WEAR paper supplementary material](https://github.com/mariusbock/wear/blob/main/supplementary_material.pdf))

## Training Data

Each inertial `.csv` file contains the following columns:
*   `sbj_id` - Unique participant id (0–21)
*   `right_arm_acc_{x,y,z}` - 3-axis acceleration at 50 Hz, right wrist
*   `right_leg_acc_{x,y,z}` - 3-axis acceleration at 50 Hz, right leg
*   `left_leg_acc_{x,y,z}` - 3-axis acceleration at 50 Hz, left leg
*   `left_arm_acc_{x,y,z}` - 3-axis acceleration at 50 Hz, left arm
*   `label ` - Activity label

Each `.npy` file contains pre-extracted video features for the corresponding participant's egocentric video. Features were extracted using [VideoMAEv2-Base](https://huggingface.co/OpenGVLab/VideoMAEv2-Base), a transformer-based video understanding model pretrained on large-scale video data. Participants can use these features as a compact, high-level representation of the visual content.
Features were extracted at 30 FPS with frames resized to `224×224` (matching the format of the pretraining data). For each frame `i`, the model takes a clip of `16` frames (frame i, 8 past frames, and 7 future frames) as input and outputs a single `768`-dimensional feature vector. Each row `i` in the `.npy` file corresponds to this feature vector for frame `i`.

## Test Data

The test set consists of `N=12,234` pre-windowed samples across three files:

* `test_inertial_data.npy`: Inertial windows at 50 Hz; shape `(N, 50, 3)` per sensor location
* `test_videomae_data.npy`: VideoMAEv2 features at 30 FPS; shape `(N, 15, 768)`. Windows contain `15` feature vectors instead of the `30` to avoid frames near the boundaries of a window to leak information about past/ future context not part of the 1 second windows.
* `test_meta_data.csv`: Window `id`, `subject_id`, and `inertial_sensor_location` for each window

All three files share the same ordering, defined by the `id` column in `test_meta_data.csv`.

## Columns Submission File & Label Encoding
To submit a solution please create a `.csv` with columns `id` and `target_value` (predicted activity label) for all test windows, using the following mapping:

*    `null`: 0,
*    `jogging`: 1,
*    `jogging (rotating arms)`: 2,
*    `jogging (skipping)`: 3,
*    `jogging (sidesteps)`: 4,
*    `jogging (butt-kicks)`: 5,
*    `stretching (triceps)`: 6,
*    `stretching (lunging)`: 7,
*    `stretching (shoulders)`: 8,
*    `stretching (hamstrings)`: 9,
*    `stretching (lumbar rotation)`: 10,
*    `push-ups`: 11,
*    `push-ups (complex)`: 12,
*    `sit-ups`: 13,
*    `sit-ups (complex)`: 14,
*    `burpees`: 15,
*    `lunges`: 16,
*    `lunges (complex)`: 17,
*    `bench-dips`: 18


---

## Prizes

We will announce winners of the challenge at the HASCA workshop. In order to be eligible for prizes and be part of the final ranking announced at the conference, participants need to submit and present their technical report at the HASCA workshop.

Prizes will be awarded to the **three** teams with the highest private leaderboard score **at the time of the paper submission deadline**. 

- 🥇 **1st place:** €300
- 🥈 **2nd place:** €150
- 🥉 **3rd place:** €75

**Please note that prize amounts will be awarded either via bank transfer or an equivalent gift card/voucher, depending on logistical constraints.**


---

## rules

##ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES.

**[See Section 3.18 for defined terms](rules#18.-terms)**

*The Competition named below is a skills-based competition to promote and further the field of data science. You must register via the Competition Website to enter. To enter the Competition, you must agree to these Official Competition Rules, which incorporate by reference the provisions and content of the Competition Website and any Specific Competition Rules herein (collectively, the "Rules"). Please read these Rules carefully before entry to ensure you understand and agree. You further agree that Submission in the Competition constitutes agreement to these Rules. You may not submit to the Competition and are not eligible to receive the prizes associated with this Competition unless you agree to these Rules. These Rules form a binding legal agreement between you and the Competition Sponsor with respect to the Competition. Your competition Submissions  must conform to the requirements stated on the Competition Website. Your Submissions will be scored based on the evaluation metric described on the Competition Website. Subject to compliance with the Competition Rules, Prizes, if any, will be awarded to Participants with the best scores, based on the merits of the data science models submitted. See below for the complete Competition Rules.*

**You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts.**

<h3>1. COMPETITION-SPECIFIC TERMS</h3>
<h4>1. COMPETITION TITLE</h4> 2nd WEAR Dataset Challenge @HASCA 2025
<h4>2. COMPETITION WEBSITE</h4> https://www.kaggle.com/competitions/second-wear-dataset-challenge
<h4>3. PRIZES</h4>
First Prize: tba
Second Prize: tba
Third Prize: tba
<h4>4. DATA LICENSE: CC BY-NC-SA 4.0

###2. COMPETITION-SPECIFIC RULES 
In addition to the provisions of the General Competition Rules below, you understand and agree to these Competition-Specific Rules required by the Competition Sponsor:

####1. TEAM LIMITS
a. The maximum Team size is ten (10).</h5>
b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running.</h5>

####2. SUBMISSION LIMITS
a. You may submit a maximum of five (5) Submissions per day.</h5>
b. You may select one (1) Final Submission for judging.</h5>

####3. COMPETITION TIMELINE
a. Competition Timeline dates (including Entry Deadline, Technical Report and Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are reflected on the competition’s Overview page.</h5>

####4. COMPETITION DATA

WEAR is offered under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. You are free to use, copy, and redistribute the material for non-commercial purposes provided you give appropriate credit, provide a link to the license, and indicate if changes were made. If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original. You may not use the material for commercial purposes.

####5. ELIGIBILITY &  WINNER’S OBLIGATIONS

a. You do not work in or collaborate with the WEAR dataset project (<a href=http://mariusbock.github.io/wear/>http://mariusbock.github.io/wear/"</a>)
b. If you submit an entry, but are not qualified to enter the contest, this entry is voluntary. The organizers reserve the right to evaluate it for scientific purposes. If you are not qualified to submit a contest entry and still choose to submit one, under no circumstances will such entries qualify for sponsored prizes.
c. As a condition to being awarded a Prize, a Prize winner must publish a technical report in the proceedings of the <a href="http://hasca2024.hasc.jp/">HASCA workshop</a>. Publishing of the paper also requires that at least one team member is registered for the <a href="http://hasca2024.hasc.jp/">HASCA workshop</a>.
d. Submission of each team's final predictions will happen via Kaggle. Only one single submission is allowed per team. The same person cannot be in multiple teams, except if that person is a supervisor. 
e. The final ranking of teams eligible for prizes, will be determined based on the private leaderboard score as of the technical report submission deadline. When submitting their technical report, participants need to tell organizers their team name as specified on Kaggle, and which submission on Kaggle is going to be their submission used for ranking. If authors do not tell us specifically which solution is to be used, we will use their latest submission as of the paper submission deadline.
f.  Prizes will be awarded to the three teams with the highest private leaderboard score at the time of the paper submission deadline.


---

## Citation

```
@article{bock2024wear,
    author = {Bock, Marius and Kuehne, Hilde and Van Laerhoven, Kristof and Moeller, Michael},
    title = {WEAR: An Outdoor Sports Dataset for Wearable and Egocentric Activity Recognition},
    year = {2024},
    volume = {8},
    number = {4},
    journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. (IMWUT)},
    numpages = {21},
    articleno = {175},
    doi = {10.1145/3699776},
    url={https://dl.acm.org/doi/10.1145/3699776}
}
```


---

## Contact

If you have any questions feel free to contact me via [e-mail](mailto:marius.bock@uni-siegen.de) or use the [discussion forum](https://www.kaggle.com/competitions/second-wear-dataset-challenge/discussion/) here on Kaggle.


---

## foundational-rules

The following Kaggle Competition Foundational Rules (“ Foundational Rules ”) apply to every competition regardless of whether the Sponsor creates competition-specific rules. Any competition-specific rules provided by the Sponsor are in addition to these rules, and in the case of any conflict or inconsistency, these Foundational Rules control and nullify contrary competition-specific rules.
###GENERAL COMPETITION RULES - BINDING AGREEMENT
####1. ELIGIBILITY
a. To be eligible to enter the Competition, you must be:</h5>
1. a registered account holder at Kaggle.com; </h6>
2. the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor); </h6>
3. not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, or North Korea; and</h6>
4. not a person or representative of an entity under U.S. export controls or sanctions (see: [https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx][1]).</h6>

b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.</h5>

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.</h5>   

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.</h5>

####2. SPONSOR AND HOSTING PLATFORM

a. The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at [www.kaggle.com/terms][2] in addition to these Rules.</h5>

####3. COMPETITION PERIOD
a. For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the “Competition Period”). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.</h5>

####4. COMPETITION ENTRY
a. NO PURCHASE NECESSARY TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize.</h5>
b. Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.</h5>
c. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible.</h5>
d. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements. </h5>

####5. INDIVIDUALS AND TEAMS
a. Individual Account. You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website. </h5>
b. Teams. If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website.</h5>
c. Team Merger. Teams may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules. </h5>
d. Private Sharing. No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.</h5>

####6. SUBMISSION CODE REQUIREMENTS
a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition (“Competition Code”). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification.</h5>
b. Public Code Sharing. You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see [www.opensource.org][3]) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code.</h5>
c. Use of Open Source. Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see [www.opensource.org][4]) that in no event limits commercial use of such code or model containing or depending on such code.</h5>

####7. DETERMINING WINNERS
a. Each Submission will be scored and ranked by the evaluation metric stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set.</h5>
b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner.</h5>

####8. NOTIFICATION OF WINNERS & DISQUALIFICATION
a. The potential winner(s) will be notified by email.</h5> 
b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition’s judging criteria.</h5>
c. In case (i) and (ii) above Kaggle may disqualify the Participant.  However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant’s compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph.</h5>
d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle.</h5>
e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded.</h5>
f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.</h5>

####9. PRIZES
a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants. </h5>
b. All Prizes are subject to Competition Sponsor's review and verification of the Participant’s eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions).</h5>
c. A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8.</h5>
d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed. </h5>
e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above.</h5>
f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.</h5>

####10. TAXES
a. ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.</h5>

####11. GENERAL CONDITIONS
a. All federal, state, provincial and local laws and regulations apply.</h5>

####12. PUBLICITY
a. You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.</h5>

####13. PRIVACY
a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address (“Personal Information”). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <[www.kaggle.com/privacy][6]>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <[www.kaggle.com/contact][7]>.</h5>
b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.</h5>

####14. WARRANTY, INDEMNITY AND RELEASE 
a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses.  You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law.</h5>
b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (b) any misrepresentation made by you in connection with the Competition; (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity.</h5>
c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.</h5>

####15. INTERNET
a. Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant’s ability to participate.</h5>

####16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY
a. If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website.  Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.</h5>

####17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT
a. Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.</h5>

####18. DEFINITIONS
a. "Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants. </h5>
b. An “Entry” is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition.</h5>
c. A “Final Submission” is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard.</h5>
d. A “Participant” or “Participant User” is an individual who participates in a competition by entering the competition and making a Submission.</h5>
e. The “Private Leaderboard” is a ranked display of Participants’ Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition.</h5>
f. The “Public Leaderboard” is a ranked display of Participants’ Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition.</h5>
g. A “Sponsor” is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules.</h5>
h. A “Submission” is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor.</h5>
i. A “Team” is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.</h5>

  [1]: https://www.treasury.gov/resource-center/sanctions/Programs/Pages/Programs.aspx
  [2]: http://www.kaggle.com/terms
  [3]: http://www.opensource.org
  [4]: http://www.opensource.org
  [5]: https://www.kaggle.com/WinningModelDocumentationGuidelines
  [6]: http://www.kaggle.com/privacy
  [7]: http://www.kaggle.com/contact

