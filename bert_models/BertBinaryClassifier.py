class DistillBERTClasBinary(torch.nn.Module):
  def __init__(self):
        super(DistillBERTClassBinary, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)

  def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
        
  def freeze_bert_encoder(self):
        for param in self.l1.parameters():
            param.requires_grad = False
    
  def unfreeze_bert_encoder(self):
        for param in self.l1.parameters():
            param.requires_grad = True