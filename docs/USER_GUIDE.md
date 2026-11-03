# User Guide

ContractIQ helps you review contracts faster. It is an informational aid, not legal advice, so check important points against the contract itself.

## Getting started

Three fictional contracts are available in `data/sample_contracts/` for you to try:

- `Vendor_Service_Agreement.pdf`
- `Software_Licensing_Agreement.pdf`
- `Employment_Agreement.pdf`

Use the sidebar to move between the four sections.

## Dashboard

The Dashboard shows:

- Total uploaded contracts
- Total indexed chunks, which are the searchable passages
- Questions asked in this session
- Your recent documents
- Buttons for moving to other sections

## Uploading documents

1. Open **Document Library**.
2. Drag PDFs into the upload box, or select **Browse files**.
3. Select **Process documents**.

The results appear on the page.

| Message | Meaning |
|---|---|
| Indexed | The contract is ready |
| Already processed | The same content has already been uploaded |
| Failed | The reason is shown, and other files are not affected |

The first upload takes longer because the embedding model (about 90 MB) has to download.

**Supported files:** text-based PDFs up to 25 MB. Scanned PDFs, password-protected PDFs, Word files and Excel files are not supported.

## Document library

Each document shows its status, page count and chunk count.

- **Selected:** ticked documents are searched by the AI Assistant.
- **Remove:** deletes the document from the library, the search index and local storage.

Your documents remain available after you restart the app. Chat history does not.

## Contract summaries

1. Open **Contract Intelligence**.
2. Select a contract.
3. Select **Analyse contract**.

The analysis makes three AI requests, so it may take a minute. The **Executive summary** gives a short overview with page references.

## Clause extraction

**Extracted clauses** shows nine fields:

1. Contract Title
2. Contracting Parties
3. Effective Date
4. Contract Duration
5. Payment Terms
6. Renewal Conditions
7. Termination Conditions
8. Confidentiality Obligations
9. Penalty or Liability Clauses

Each field has a status:

| Status | Meaning |
|---|---|
| Found | ContractIQ confirmed the supporting text in the document |
| Not found | No information was identified |
| Needs review | A value was returned, but ContractIQ could not fully confirm it |

"Not found" does not prove that a term is absent, so check anything important yourself.

## Risk identification

**Potential risk findings** highlights clauses that may deserve human review. This means "look at this", not "this is a problem".

| Level | Meaning |
|---|---|
| High Attention | Could cause significant financial or operational exposure |
| Medium Attention | One-sided or unclear terms worth clarifying |
| Low Attention | Minor points worth noting |

Each finding shows the issue, why it deserves attention, the source document and the page. Missing items are labelled as not stated in the document. If a finding says **Not verified**, ContractIQ could not find the supporting text, so treat it with caution.

An empty list does not mean that a contract is safe.

## AI chatbot

1. Open **AI Assistant**.
2. Check the list of documents being searched.
3. Type a question and press Enter.

Select **Clear chat** to erase the conversation.

Each question is answered independently, because the model does not see earlier questions. Instead of "What about the second one?", ask "What is the termination notice period in the employment agreement?"

## Source references

Expand **Sources** under an answer to see the filename, page and supporting text. A label such as `[S1]` in the answer links to the matching source.

- **"This information was not found"** means that the selected documents did not provide a supported answer.
- **"This answer did not cite the contract"** means that you should treat the answer as unverified. The passages that were searched are listed under **Retrieved passages**.

## Sample questions

**Vendor Service Agreement**

- What is the monthly service fee?
- How many days does the client have to pay an invoice?
- How much notice is needed to stop the agreement renewing?
- What happens if the vendor misses its service level?
- Can the client terminate for convenience?

**Software Licensing Agreement**

- What is the annual licence fee?
- What happens if we go over our licensed user count?
- Can we terminate early?
- What is the cap on the licensor's liability?
- Does the agreement say how our data will be returned?

**Employment Agreement**

- What is the notice period after probation?
- When must the joining bonus be repaid?
- Is there a non-compete restriction?
- How many days of paid leave are provided?
- What are the liquidated damages?

**To test missing information**

- What is the arbitrator's hourly rate?
- Is there a parental leave policy?

## Protecting confidential information

Contract text is sent to Google's Gemini API for processing. Google's pricing page says that free-tier usage may be used to improve its products. Use the sample contracts or non-confidential documents with a free-tier key.

Text from uploaded documents is also stored unencrypted in `data/vector_store/` on your computer until you remove the document.
